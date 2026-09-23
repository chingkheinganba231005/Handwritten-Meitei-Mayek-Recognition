"""Training loop, weight averaging and prediction.

A member config (see ``model.MEMBERS``) may also set: ``meta`` (size-aware),
``pretrained``, ``seed``, ``smoothing``, ``ema``, ``batch``, ``aug`` (a preset
name from ``augment.AUG_PRESETS``) and ``keep_raw`` (also save the last raw,
non-averaged weights).
"""

import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .augment import AUG_PRESETS, augment, view
from .model import make_model


def amp_dtype(device):
    return torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else None


class EMA:
    """Exponential moving average of the weights; this copy is what gets evaluated and kept."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay, self.updates = decay, 0
        for p in self.module.parameters():
            p.requires_grad_(False)
        # state_dict tensors share storage with the live weights, so these lists stay valid
        pairs = list(zip(self.module.state_dict().values(), model.state_dict().values()))
        self.floats = [list(t) for t in zip(*[(e, m) for e, m in pairs if e.dtype.is_floating_point])]
        self.others = [(e, m) for e, m in pairs if not e.dtype.is_floating_point]

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        torch._foreach_lerp_(self.floats[0], self.floats[1], 1 - d)
        for e, m in self.others:
            e.copy_(m)


def param_groups(model, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def predict(model, idx, cfg, store, views=("id",), bs=512):
    """Softmax probabilities, averaged over the given test-time views."""
    model.eval()
    amp = amp_dtype(store.device)
    idx = np.asarray(idx)
    out = []
    for i in range(0, len(idx), bs):
        chunk = idx[i:i + bs]
        x, meta = store.fetch(chunk, cfg["channels"]), store.meta(chunk, cfg)
        p = 0
        for v in views:
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                logits = model(store.normalize(view(x, v), cfg["channels"]), meta)
            p = p + logits.float().softmax(-1)
        out.append((p / len(views)).cpu())
    return torch.cat(out).numpy()


def load_model(path, cfg=None, num_classes=55, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = cfg or ck["cfg"]
    model = make_model(cfg, num_classes, pretrained=False, device=device)
    model.load_state_dict(ck["model"])
    return model.eval()


def train(name, cfg, train_idx, tag, store, labels, work, val_idx=None, num_classes=55, log=print):
    """Train one model and return its averaged copy. Resumes from ``work`` after a disconnect."""
    device = store.device
    out = Path(work) / "runs" / name / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.pt").exists():
        return load_model(out / "final.pt", cfg, num_classes, device)

    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    random.seed(seed)
    aug = AUG_PRESETS[cfg.get("aug", "full")]
    amp = amp_dtype(device)
    train_idx = np.asarray(train_idx)
    bs = min(cfg.get("batch", 256), len(train_idx))
    steps_per_epoch = max(1, len(train_idx) // bs)
    total = cfg["epochs"] * steps_per_epoch
    warm = max(1, min(3 * steps_per_epoch, total // 5))

    def lr_at(step):
        if step < warm:
            return cfg["lr"] * (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return cfg["lr"] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))

    model = make_model(cfg, num_classes, device=device)
    ema = EMA(model, cfg.get("ema", 0.9995))
    opt = torch.optim.AdamW(param_groups(model, cfg["wd"]), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.get("smoothing", 0.1))
    y_all = torch.as_tensor(labels, device=device)
    start, step, history = 1, 0, []
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.module.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        ema.updates, start, step, history = ck["ema_updates"], ck["epoch"] + 1, ck["step"], ck["history"]
        log(f"{name}/{tag}: resuming at epoch {start}")

    for epoch in range(start, cfg["epochs"] + 1):
        model.train()
        t0, running = time.time(), torch.zeros((), device=device)
        gen = torch.Generator().manual_seed(seed * 1000 + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=gen).numpy()]
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            x = store.normalize(augment(store.fetch(idx, cfg["channels"]), aug), cfg["channels"])
            y = y_all[torch.as_tensor(idx, device=device)]
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                loss = crit(model(x, store.meta(idx, cfg)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ema.update(model)
            running += loss.detach()
            step += 1

        row = {"epoch": epoch, "loss": running.item() / steps_per_epoch, "lr": lr_at(step - 1),
               "sec": round(time.time() - t0)}
        if val_idx is not None and len(val_idx) and (epoch % 5 == 0 or epoch == cfg["epochs"]):
            row["val_ema"] = float((predict(ema.module, val_idx, cfg, store).argmax(1) == labels[val_idx]).mean())
        history.append(row)
        log(f"{name}/{tag} " + "  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
        torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
                    "ema_updates": ema.updates, "epoch": epoch, "step": step, "history": history}, out / "last.pt")

    if cfg.get("keep_raw"):
        torch.save({"model": model.state_dict(), "cfg": cfg}, out / "raw.pt")
    torch.save({"model": ema.module.state_dict(), "cfg": cfg, "history": history}, out / "final.pt")
    (out / "last.pt").unlink(missing_ok=True)
    return ema.module.eval()


def cached_preds(work, tag, split, make):
    """Load saved probabilities (float16 on disk) or compute and save them."""
    f = Path(work) / "preds" / f"{tag}_{split}.npy"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, make().astype(np.float16))
    return np.load(f).astype(np.float32)
