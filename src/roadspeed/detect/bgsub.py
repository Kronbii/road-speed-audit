"""Background-subtraction detector for fixed cameras (OpenCV MOG2).

Foreground pixels (shadow pixels excluded) are cleaned with morphology, grouped
into connected components, and nearby fragments of one vehicle are merged. It
needs no model download and no GPU, which suits an offline-first pilot; its known
weaknesses are merged blobs for vehicles that overlap in the image, slow
adaptation to sudden lighting changes, and parked vehicles fading into the
background.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import Detection, touches_border


@dataclass
class BGSubConfig:
    history: int = 500
    var_threshold: float = 25.0
    detect_shadows: bool = True
    shadow_threshold: float = 0.5  # MOG2 "tau": darker-than-background ratio counted as shadow
    # "merge": shadow-labelled pixels connected to real foreground are kept (grey vehicles
    # on grey asphalt are otherwise labelled "shadow"); "remove": MOG2 default, shadow
    # pixels are background. Cast shadows lie on the road and move rigidly with the
    # vehicle, so keeping them does not bias speed, but can merge neighbouring blobs.
    shadow_mode: str = "merge"
    min_core_frac: float = 0.08  # in "merge" mode, a blob needs this share of non-shadow pixels
    learning_rate: float = -1.0  # -1 = automatic
    min_area_frac: float = 1.5e-4  # minimum blob area as a fraction of the frame area
    open_frac: float = 3.0 / 720  # opening kernel size as a fraction of frame height
    close_frac: float = 9.0 / 720
    merge_gap_frac: float = 6.0 / 720  # merge fragments closer than this (fraction of height)
    high_score_area: float = 4.0  # blobs >= this * min_area get score 1
    # Sub-pixel refinement of the bottom edge at the 50 % level of the difference to
    # the background model. The foreground threshold sits in the tail of a blurred
    # edge, which places the raw blob boundary a few pixels outside the object.
    refine_bottom: bool = True
    background_every: int = 10  # frames between refreshes of the background image


class BackgroundSubtractionDetector:
    def __init__(
        self,
        cfg: BGSubConfig,
        frame_size: tuple[int, int],
        roi_mask: np.ndarray | None = None,
    ):
        self.cfg = cfg
        self.width, self.height = frame_size
        self.roi_mask = roi_mask
        self.mog = cv2.createBackgroundSubtractorMOG2(
            history=cfg.history, varThreshold=cfg.var_threshold, detectShadows=cfg.detect_shadows
        )
        if cfg.detect_shadows:
            self.mog.setShadowThreshold(cfg.shadow_threshold)
        h = self.height
        self.min_area = max(12, int(round(cfg.min_area_frac * self.width * self.height)))

        def odd(x):
            k = max(1, int(round(x * h)))
            return k + 1 - k % 2

        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(cfg.open_frac),) * 2)
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(cfg.close_frac),) * 2)
        self.merge_gap = max(1, int(round(cfg.merge_gap_frac * h)))
        self.last_mask: np.ndarray | None = None
        self._bg: np.ndarray | None = None
        self._n = 0
        # process only the bounding rectangle of the region of interest
        if roi_mask is not None and roi_mask.any():
            x, y, w, hh = cv2.boundingRect(roi_mask.astype(np.uint8))
            self.crop = (
                max(0, x - 4),
                max(0, y - 4),
                min(self.width, x + w + 4),
                min(self.height, y + hh + 4),
            )
        else:
            self.crop = (0, 0, self.width, self.height)
        x0, y0, x1, y1 = self.crop
        self._roi = None if roi_mask is None else roi_mask[y0:y1, x0:x1].astype(np.uint8)
        self._sh = y1 - y0

    def _sub(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.crop
        return frame[y0:y1, x0:x1]

    def prime(self, frames) -> None:
        """Initialise the background model with the per-pixel median of ``frames``.

        Using a median of frames spread over the clip gives a clean background even
        when traffic is present from the first frame.
        """
        frames = list(frames)
        if not frames:
            return
        med = np.median(np.stack([self._sub(f) for f in frames]), axis=0).astype(np.uint8)
        self.mog.apply(med, learningRate=1.0)
        for _ in range(10):
            self.mog.apply(med, learningRate=0.1)

    def foreground(self, frame: np.ndarray) -> np.ndarray:
        m = self.mog.apply(frame, learningRate=self.cfg.learning_rate)
        if self.cfg.detect_shadows and self.cfg.shadow_mode == "merge":
            fg = (m > 0).astype(np.uint8)
        else:
            fg = (m == 255).astype(np.uint8)
        if self._roi is not None:
            fg &= self._roi
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)
        self.last_mask = fg
        self.last_core = m == 255
        return fg

    def _background(self) -> np.ndarray | None:
        if self._bg is None or self._n % max(1, self.cfg.background_every) == 0:
            bg = self.mog.getBackgroundImage()
            self._bg = None if bg is None else bg.astype(np.int16)
        return self._bg

    def refine_bottom_edge(self, frame: np.ndarray, bbox) -> float:
        """Sub-pixel row of the bottom edge (50 % crossing of |frame - background|)."""
        bg = self._background()
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        if bg is None or y2 - y1 < 6:
            return float(y2)
        w = x2 - x1 + 1
        c0, c1 = x1 + w // 4, x2 - w // 4 + 1
        r0, r1 = max(y1, y2 - 8), min(self._sh, y2 + 5)
        if c1 <= c0 or r1 - r0 < 6:
            return float(y2)
        D = (
            np.abs(frame[r0:r1, c0:c1].astype(np.int16) - bg[r0:r1, c0:c1])
            .sum(axis=2)
            .astype(float)
        )
        yb = y2 - r0
        inner = D[: max(1, yb - 1)]
        outer = D[yb + 2 :] if yb + 2 < D.shape[0] else D[-1:]
        lin = np.percentile(inner, 75, axis=0)
        lout = np.median(outer, axis=0)
        T = 0.5 * (lin + lout)
        edges = []
        for j in range(D.shape[1]):
            if lin[j] - lout[j] < 12:  # too little contrast in this column
                continue
            col = D[:, j]
            above = np.nonzero(col[: yb + 2] >= T[j])[0]
            if len(above) == 0:
                continue
            i = int(above[-1])
            if i + 1 >= len(col):
                continue
            d0, d1 = col[i], col[i + 1]
            frac = (d0 - T[j]) / (d0 - d1) if d0 > d1 else 0.0
            edges.append(r0 + i + min(max(frac, 0.0), 1.0))
        if len(edges) < max(2, (c1 - c0) // 4):
            return float(y2)
        e = float(np.median(edges))
        return min(max(e, float(y2) - 4.0), float(y2) + 1.0)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        full = frame
        frame = self._sub(full)
        ox, oy = self.crop[0], self.crop[1]
        fg = self.foreground(frame)
        self._n += 1
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        merge = self.cfg.detect_shadows and self.cfg.shadow_mode == "merge"
        if merge and n > 1:
            core = np.bincount(labels[self.last_core], minlength=n)
        boxes = []
        areas = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < max(4, self.min_area // 4):
                continue
            if merge and core[i] < self.cfg.min_core_frac * a:
                continue  # a shadow with no object attached
            boxes.append([x, y, x + w - 1, y + h - 1])
            areas.append(int(a))
        boxes, areas = _merge_fragments(boxes, areas, self.merge_gap)
        dets = []
        for b, a in zip(boxes, areas, strict=True):
            if a < self.min_area:
                continue
            score = float(min(1.0, a / (self.cfg.high_score_area * self.min_area)))
            bb = np.array(b, dtype=float)
            x1, y1, x2, y2 = b
            band = max(2, int(round(0.15 * (y2 - y1 + 1))))
            cols = np.nonzero(fg[max(y1, y2 - band + 1) : y2 + 1, x1 : x2 + 1].any(axis=0))[0]
            bw = float(cols[-1] - cols[0] + 1) if len(cols) else float(x2 - x1 + 1)
            if self.cfg.refine_bottom and bb[3] < self._sh - 3:
                bb[3] = self.refine_bottom_edge(frame, b)
            bb += np.array([ox, oy, ox, oy], float)
            dets.append(
                Detection(
                    bbox=bb,
                    score=score,
                    area_px=a,
                    touches_border=touches_border(bb, self.width, self.height),
                    bottom_width_px=bw,
                )
            )
        return dets


def _merge_fragments(boxes, areas, gap):
    """Greedily merge boxes that overlap or nearly touch (fragments of one vehicle)."""
    boxes = [list(b) for b in boxes]
    areas = list(areas)
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if (
                    a[0] - gap <= b[2]
                    and b[0] - gap <= a[2]
                    and a[1] - gap <= b[3]
                    and b[1] - gap <= a[3]
                ):
                    # require substantial horizontal overlap so vehicles side by side
                    # in adjacent lanes are not merged just because they touch
                    ov = min(a[2], b[2]) - max(a[0], b[0])
                    if ov >= 0.5 * min(a[2] - a[0], b[2] - b[0]):
                        boxes[i] = [
                            min(a[0], b[0]),
                            min(a[1], b[1]),
                            max(a[2], b[2]),
                            max(a[3], b[3]),
                        ]
                        areas[i] += areas[j]
                        del boxes[j], areas[j]
                        changed = True
                        break
            if changed:
                break
    return boxes, areas
