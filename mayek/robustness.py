"""Synthetic corruptions of the raw scans, for the robustness evaluation.

Each corruption is applied to the original greyscale image before any
preprocessing, with a seed derived from the file path, so every run sees the
same corrupted copies.
"""

import zlib

import cv2
import numpy as np
from PIL import Image

from .preprocess import preprocess

CORRUPTIONS = {
    "noise": (0.05, 0.10),        # Gaussian noise, s.d. as a fraction of 255
    "blur": (0.6, 1.0),           # Gaussian blur sigma in pixels (images are ~24 px)
    "rotation": (10, 20),         # degrees, random sign
    "contrast": (0.5, 0.25),      # ink contrast kept
    "shading": (0.3, 0.6),        # linear illumination fall-off across the image
    "salt_pepper": (0.02, 0.05),  # fraction of pixels flipped to black or white
    "low_res": (16, 12),          # downsample to this many pixels, then back up
}


def corrupt(gray, kind, level, rng):
    g = gray.astype(np.float32)
    H, W = g.shape
    paper = float(np.median(g))
    if kind == "noise":
        g = g + rng.normal(0, level * 255, g.shape)
    elif kind == "blur":
        g = cv2.GaussianBlur(g, (0, 0), level)
    elif kind == "rotation":
        angle = level * rng.choice([-1, 1])
        M = cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), angle, 1.0)
        g = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=paper)
    elif kind == "contrast":
        g = paper + level * (g - paper)
    elif kind == "shading":
        a = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:H, 0:W]
        t = (np.cos(a) * xx / max(W - 1, 1) + np.sin(a) * yy / max(H - 1, 1))
        t = (t - t.min()) / max(np.ptp(t), 1e-6)
        g = g * (1 - level * t)
    elif kind == "salt_pepper":
        m = rng.random(g.shape)
        g = np.where(m < level / 2, 0, np.where(m > 1 - level / 2, 255, g))
    elif kind == "low_res":
        side = int(level)
        small = cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA)
        g = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(kind)
    return np.clip(g + 0.5, 0, 255).astype(np.uint8)


def corrupt_file(path, size, kind, level):
    """Read a dataset image, corrupt it, preprocess it (picklable for multiprocessing)."""
    gray = np.array(Image.open(path).convert("L"))
    rng = np.random.default_rng(zlib.crc32(f"{path}|{kind}|{level}".encode()))
    return preprocess(corrupt(gray, kind, level, rng), size)
