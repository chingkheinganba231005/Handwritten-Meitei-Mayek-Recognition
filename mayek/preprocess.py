"""Turning a character image into network input.

``preprocess`` is what every model was trained on: Otsu ink mask, ink scaled
to 1 and paper to 0, crop to the ink with a 16% margin, pad to a square,
resize, then skeleton and distance transform of the result.

``to_dataset_scale`` is only for the demo. TUMMHCD images are small scans
(mostly 24 x 24 px) with strokes two to three pixels wide. A character drawn
on a 400 px canvas or photographed with a phone looks nothing like that, so
it is first brought to the same scale: strokes thickened to the dataset's
typical width, then the whole frame shrunk to 24 px.
"""

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

MARGIN = 1.16          # crop side = 1.16 x the larger side of the ink box (+2 px)
BINARY_THRESHOLD = 0.35
DATASET_SIDE = 24      # typical TUMMHCD image size


def _to_uint8(a):
    return (a * 255 + 0.5).astype(np.uint8)


def load_gray(image):
    """Path, PIL image or array -> uint8 greyscale array (transparent pixels count as white paper)."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[2] == 4:
            image = Image.fromarray(arr.astype(np.uint8), "RGBA")
        elif arr.ndim == 3:
            return np.asarray(Image.fromarray(arr.astype(np.uint8)).convert("L"))
        else:
            return arr.astype(np.uint8)
    elif not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(paper, rgba)
    return np.asarray(image.convert("L"))


def ink_mask(gray):
    """Otsu ink mask and the greyscale with dark ink on light paper."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = th == 0
    gray = gray.astype(np.float32)
    if ink.mean() > 0.5:  # light ink on dark paper
        gray, ink = 255 - gray, ~ink
    return gray, ink


def preprocess(gray, size=128):
    """uint8 greyscale -> (ink, skeleton, distance) as uint8 size x size, plus the 5 size numbers.

    The size numbers are [image height, image width, ink box height, ink box
    width, ink fraction], taken before any cropping.
    """
    H, W = gray.shape
    gray, ink = ink_mask(gray)
    paper = np.median(gray[~ink]) if (~ink).any() else 255.0
    dark = np.percentile(gray[ink], 5) if ink.any() else 0.0
    g = np.clip((paper - gray) / max(paper - dark, 1.0), 0, 1)

    ys, xs = np.nonzero(ink)
    y1, y2, x1, x2 = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1) if len(xs) else (0, H, 0, W)
    bh, bw = y2 - y1, x2 - x1
    side = int(max(bh, bw) * MARGIN) + 2
    canvas = np.zeros((side, side), np.float32)
    oy, ox = (side - bh) // 2, (side - bw) // 2
    canvas[oy:oy + bh, ox:ox + bw] = g[y1:y2, x1:x2]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    img = np.clip(cv2.resize(canvas, (size, size), interpolation=interp), 0, 1)

    binary = img > BINARY_THRESHOLD
    skel = skeletonize(binary).astype(np.float32)
    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
    dist = dist / max(float(dist.max()), 1e-6)
    return _to_uint8(img), _to_uint8(skel), _to_uint8(dist), np.array([H, W, bh, bw, ink.mean()], np.float32)


def preprocess_file(path, size=128):
    """Dataset images, read exactly as during training."""
    return preprocess(np.array(Image.open(path).convert("L")), size)


def size_features(meta):
    """The 5 raw size numbers -> the features the size-aware models see (before standardising)."""
    meta = np.atleast_2d(meta)
    return np.column_stack([np.log(np.maximum(meta[:, :4], 1)), meta[:, 4]]).astype(np.float32)


def stroke_width(ink):
    """Median stroke width in pixels: twice the distance to the paper, measured on the skeleton."""
    if not ink.any():
        return 0.0
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    skel = skeletonize(ink)
    return float(2 * np.median(dist[skel])) if skel.any() else 0.0


def to_dataset_scale(image, crop=False, stroke_ratio=0.1, side=DATASET_SIDE):
    """Any character image -> a small scan like the TUMMHCD ones (uint8, dark ink on white).

    crop=False treats the whole image as the writing box (a drawing canvas):
    how much of it the character fills is kept. crop=True first cuts the
    character out of a larger photo with a 20% margin. Images that are already
    dataset-sized are returned unchanged.
    """
    gray = load_gray(image)
    if max(gray.shape) <= 2 * side:
        return gray
    g, ink = ink_mask(gray)
    if not ink.any():
        return cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
    if crop:
        ys, xs = np.nonzero(ink)
        pad = int(0.2 * max(np.ptp(ys), np.ptp(xs))) + 1
        y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, gray.shape[0])
        x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, gray.shape[1])
        g, ink = g[y1:y2, x1:x2], ink[y1:y2, x1:x2]

    # square frame, paper-coloured padding
    h, w = ink.shape
    n = max(h, w)
    paper = float(np.median(g[~ink])) if (~ink).any() else 255.0
    frame = np.full((n, n), paper, np.float32)
    mask = np.zeros((n, n), bool)
    oy, ox = (n - h) // 2, (n - w) // 2
    frame[oy:oy + h, ox:ox + w], mask[oy:oy + h, ox:ox + w] = g, ink

    # thicken thin pen strokes to the dataset's relative stroke width
    target = stroke_ratio * n
    have = stroke_width(mask)
    if have and have < target:
        k = int(round(target - have)) | 1
        grown = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        dark = float(np.percentile(frame[mask], 5))
        frame = np.where(grown & ~mask, dark, frame)
    small = cv2.resize(frame, (side, side), interpolation=cv2.INTER_AREA)
    return np.clip(small + 0.5, 0, 255).astype(np.uint8)
