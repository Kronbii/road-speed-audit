"""Anonymisation of image crops used for human review.

Review crops are optional and off by default. When enabled, every crop is
pixelated to a coarse grid and blurred before it is written, so that number
plates and faces behind windscreens are not legible while the vehicle's outline,
size and colour remain visible for a reviewer judging whether a box holds one
vehicle or two. Full frames are never written.
"""

from __future__ import annotations

import cv2
import numpy as np


def anonymize(img: np.ndarray, cells: int = 16, blur_frac: float = 0.06) -> np.ndarray:
    """Pixelate to ``cells`` blocks across the longer side, then blur."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img.copy()
    s = cells / max(h, w)
    small = cv2.resize(
        img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA
    )
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    k = max(3, int(round(blur_frac * max(h, w))) | 1)
    return cv2.GaussianBlur(out, (k, k), 0)


def anonymized_crop(
    frame: np.ndarray, bbox, margin: float = 0.25, out_size: int = 160, cells: int = 16
):
    """Crop ``bbox`` (with margin) from ``frame``, anonymise it, resize to ``out_size``."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    mw, mh = margin * (x2 - x1 + 1), margin * (y2 - y1 + 1)
    a, b = int(max(0, np.floor(x1 - mw))), int(max(0, np.floor(y1 - mh)))
    c, d = int(min(W, np.ceil(x2 + mw) + 1)), int(min(H, np.ceil(y2 + mh) + 1))
    crop = frame[b:d, a:c]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), np.uint8)
    anon = anonymize(crop, cells=cells)
    h, w = anon.shape[:2]
    s = out_size / max(h, w)
    anon = cv2.resize(
        anon, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_NEAREST
    )
    canvas = np.full((out_size, out_size, 3), 40, np.uint8)
    oy, ox = (out_size - anon.shape[0]) // 2, (out_size - anon.shape[1]) // 2
    canvas[oy : oy + anon.shape[0], ox : ox + anon.shape[1]] = anon
    return canvas
