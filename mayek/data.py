"""Split index, preprocessed image cache and the tensors the training loop reads from."""

import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import CHANNELS  # noqa: F401  (re-exported)
from .preprocess import preprocess_file, size_features


def load_index(split_dir):
    """Split CSVs -> (index frame, labels, {split: row indices}). Row order is train, val, test."""
    split_dir = Path(split_dir)
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(split_dir / f"{split}.csv")
        df["split"] = split
        df["file"] = [str((split_dir / p).resolve()) for p in df.path]
        frames.append(df)
    index = pd.concat(frames, ignore_index=True)
    labels = index.label.to_numpy().copy()
    idx = {s: np.flatnonzero(index.split.to_numpy() == s) for s in ("train", "val", "test")}
    idx["full"] = np.concatenate([idx["train"], idx["val"]])
    return index, labels, idx


def _run(job):
    fn, path, size = job
    return fn(path, size)


def build_cache(files, size, cache_dir, fn=preprocess_file, tag=None, workers=None):
    """Preprocess every image once; returns (ink, skeleton, distance, size numbers) arrays."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or str(size)
    paths = [cache_dir / f"{name}_{tag}.npy" for name in ("gray", "skel", "dist", "meta")]
    if all(p.exists() for p in paths):
        return [np.load(p) for p in paths]
    t0, n = time.time(), len(files)
    arrays = [np.empty((n, size, size), np.uint8) for _ in range(3)] + [np.empty((n, 5), np.float32)]
    jobs = [(fn, f, size) for f in files]
    with get_context("fork").Pool(workers or os.cpu_count()) as pool:
        for i, out in enumerate(pool.imap(_run, jobs, chunksize=64)):
            for arr, value in zip(arrays, out):
                arr[i] = value
    for p, arr in zip(paths, arrays):
        np.save(p, arr)
    print(f"cached {n} images at {size} px in {time.time() - t0:.0f} s")
    return arrays


class Store:
    """Cached images as tensors (on the GPU if it has room), plus normalisation statistics.

    ``stats`` and ``meta_norm`` default to values computed from ``train_idx``;
    pass the training ones when wrapping other images (e.g. corrupted test copies).
    """

    def __init__(self, arrays, device, train_idx=None, stats=None, meta_norm=None, keep_on=None):
        gray, skel, dist, meta = arrays
        self.device = device
        big_gpu = device == "cuda" and torch.cuda.get_device_properties(0).total_memory > 30e9
        self.where = keep_on or (device if big_gpu else "cpu")
        self.gray, self.skel, self.dist = (torch.from_numpy(a).to(self.where) for a in (gray, skel, dist))
        self.size = gray.shape[-1]
        feats = size_features(meta)
        if meta_norm is None:
            mu, sd = feats[train_idx].mean(0), feats[train_idx].std(0) + 1e-6
            meta_norm = (mu, sd)
        self.meta_norm = meta_norm
        self.meta_z = torch.tensor((feats - meta_norm[0]) / meta_norm[1], dtype=torch.float32, device=device)
        self.stats = stats or self._stats(train_idx)

    def _stats(self, train_idx):
        rng = np.random.default_rng(0)
        sample = torch.as_tensor(rng.choice(train_idx, min(5000, len(train_idx)), replace=False))
        stats = {}
        for name, chans in (("gray", [self.gray]), ("topo", [self.gray, self.skel, self.dist])):
            x = torch.stack([c[sample.to(self.where)].float() / 255 for c in chans], 1)
            stats[name] = (x.mean((0, 2, 3)).view(1, -1, 1, 1).to(self.device),
                           x.std((0, 2, 3)).view(1, -1, 1, 1).to(self.device))
        return stats

    def fetch(self, idx, channels):
        i = torch.as_tensor(np.asarray(idx), device=self.where)
        x = self.gray[i][:, None] if channels == "gray" else torch.stack([self.gray[i], self.skel[i], self.dist[i]], 1)
        return x.to(self.device, non_blocking=True).float().div_(255)

    def normalize(self, x, channels):
        mean, std = self.stats[channels]
        return ((x - mean) / std).contiguous(memory_format=torch.channels_last)

    def meta(self, idx, cfg):
        return self.meta_z[torch.as_tensor(np.asarray(idx), device=self.device)] if cfg.get("meta") else None

    def export_stats(self):
        """JSON-friendly normalisation constants for inference."""
        out = {k: {"mean": m.flatten().tolist(), "std": s.flatten().tolist()} for k, (m, s) in self.stats.items()}
        out["meta"] = {"mean": np.asarray(self.meta_norm[0]).tolist(), "std": np.asarray(self.meta_norm[1]).tolist()}
        return out
