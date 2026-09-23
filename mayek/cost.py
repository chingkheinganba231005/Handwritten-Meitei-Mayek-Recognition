"""Model size, FLOPs and latency measurements."""

import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .model import CHANNELS


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def gflops(model, cfg, img=128):
    """GFLOPs (2 x multiply-accumulates) for one image and one view, as counted by PyTorch."""
    x = torch.zeros(1, CHANNELS[cfg["channels"]], img, img, device=next(model.parameters()).device)
    meta = torch.zeros(1, 5, device=x.device) if cfg.get("meta") else None
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model.eval()(x, meta)
    return fc.get_total_flops() / 1e9


@torch.no_grad()
def latency(model, cfg, batch, views=1, img=128, device="cpu", amp=None, repeats=20, warmup=5):
    """Median seconds per forward of `batch` images x `views` views."""
    model = model.eval().to(device)
    x = torch.rand(batch * views, CHANNELS[cfg["channels"]], img, img, device=device).contiguous(
        memory_format=torch.channels_last)
    meta = torch.zeros(batch * views, 5, device=device) if cfg.get("meta") else None
    times = []
    for i in range(warmup + repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            model(x, meta)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return float(np.median(times))
