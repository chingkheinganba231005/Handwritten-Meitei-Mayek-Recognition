import zipfile

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


def fake_zip(path, per_class=6, classes=3):
    with zipfile.ZipFile(path, "w") as z:
        for split, n in (("train", per_class), ("test", 2)):
            for c in range(classes):
                for i in range(n):
                    ok, buf = cv2.imencode(".png", glyph(c + i, thickness=1 + i % 3))
                    z.writestr(f"TUMMHCD-TEST-TRAIN/TUMMHCD{split}/{split}_{c:03d}/img{i}.png", buf.tobytes())
