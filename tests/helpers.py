import cv2
import numpy as np


def glyph(kind=0, size=24, thickness=2, scale=1.0):
    """A dark synthetic 'character' on light paper."""
    img = np.full((size, size), 235, np.uint8)
    c = size // 2
    r = int(size * 0.35 * scale)
    if kind % 3 == 0:
        cv2.circle(img, (c, c), r, 20, thickness)
    elif kind % 3 == 1:
        cv2.line(img, (c - r, c - r), (c + r, c + r), 20, thickness)
        cv2.line(img, (c - r, c + r), (c + r, c - r), 20, thickness)
    else:
        cv2.rectangle(img, (c - r, c - r // 2), (c + r, c + r), 20, thickness)
    return img
