"""Inference for released models: any character image in, top-k classes out.

A model folder holds ``config.json`` and one state dict per member, as
written by :func:`export`. Runs on CPU; no training code or data needed.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .augment import TTA_VIEWS, view
from .charset import CLASSES
from .files import must_write
from .model import make_model
from .preprocess import preprocess, size_features, to_dataset_scale


@dataclass
class Prediction:
    index: int
    prob: float

    @property
    def cls(self):
        return CLASSES[self.index]


class Recognizer:
    def __init__(self, model_dir, device="cpu"):
        self.dir = Path(model_dir)
        self.config = json.loads((self.dir / "config.json").read_text())
        self.device = device
        self.img = self.config["img"]
        stats = self.config["stats"]
        self.stats = {k: (torch.tensor(v["mean"]).view(1, -1, 1, 1), torch.tensor(v["std"]).view(1, -1, 1, 1))
                      for k, v in stats.items() if k != "meta"}
        self.meta_norm = (np.array(stats["meta"]["mean"], np.float32), np.array(stats["meta"]["std"], np.float32))
        self.members = []
        for m in self.config["members"]:
            model = make_model(m["cfg"], self.config.get("num_classes", 55), pretrained=False, device=device)
            state = torch.load(self.dir / m["file"], map_location=device, weights_only=True)
            model.load_state_dict(state)
            self.members.append((model.eval(), m))

    def dataset_like(self, image, source="canvas"):
        """The small scan the models will see (see preprocess.to_dataset_scale)."""
        return to_dataset_scale(image, crop=source == "upload", stroke_ratio=self.config.get("stroke_ratio", 0.1))

    def _inputs(self, grays):
        out = [preprocess(g, self.img) for g in grays]
        img, skel, dist = (torch.from_numpy(np.stack([o[i] for o in out])).float().div_(255) for i in range(3))
        meta = size_features(np.stack([o[3] for o in out]))
        meta = torch.from_numpy((meta - self.meta_norm[0]) / self.meta_norm[1]).float()
        return {"gray": img[:, None], "topo": torch.stack([img, skel, dist], 1)}, meta, img

    @torch.no_grad()
    def probs(self, grays, tta=True):
        """Ensemble probabilities for dataset-like uint8 greyscale images."""
        x, meta, _ = self._inputs(grays)
        total = 0
        for model, m in self.members:
            ch = m["cfg"]["channels"]
            mean, std = self.stats[ch]
            views = m["views"] if tta else ["id"]
            p = 0
            for v in views:
                inp = ((view(x[ch], v) - mean) / std).contiguous(memory_format=torch.channels_last).to(self.device)
                logits = model(inp, meta.to(self.device) if m["cfg"].get("meta") else None)
                p = p + logits.float().softmax(-1)
            total = total + m["weight"] * p / len(views)
        return total.cpu().numpy()

    def predict(self, image, source="canvas", tta=True, topk=5):
        """Returns (top-k predictions, the 128 px ink image the models saw)."""
        small = self.dataset_like(image, source)
        p = self.probs([small], tta)[0]
        top = np.argsort(-p)[:topk]
        _, _, img = self._inputs([small])
        return [Prediction(int(i), float(p[i])) for i in top], (img[0].numpy() * 255).astype(np.uint8)


def export(members, run_dir, out_dir, stats, img=128, stroke_ratio=0.1, num_classes=55):
    """Write a model folder for Recognizer.

    members: list of dicts with name, cfg, views and weight; the averaged
    weights are read from ``run_dir/<name>/full/final.pt``.
    stats: Store.export_stats() of the training data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for m in members:
        ck = torch.load(Path(run_dir) / m["name"] / "full" / "final.pt", map_location="cpu", weights_only=False)
        must_write(lambda f: torch.save(ck["model"], f), out_dir / f"{m['name']}.pt")
        cfg = {k: v for k, v in m["cfg"].items() if k in ("arch", "channels", "drop_path", "meta")}
        entries.append({"name": m["name"], "file": f"{m['name']}.pt", "cfg": cfg,
                        "views": list(m["views"]), "weight": float(m["weight"])})
    config = {"img": img, "num_classes": num_classes, "stroke_ratio": float(stroke_ratio),
              "stats": stats, "members": entries, "tta_views": TTA_VIEWS}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    return out_dir
