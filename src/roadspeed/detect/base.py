"""Common detection types and the backend factory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """One detected object in one frame (pixel box ``x1, y1, x2, y2``, inclusive)."""

    bbox: np.ndarray
    score: float
    label: str = "vehicle"
    area_px: int = 0
    touches_border: bool = False
    bottom_width_px: float | None = None  # width of the blob's lowest rows (merge indicator)

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return 0.5 * (x1 + x2), float(y2)


def touches_border(bbox, width: int, height: int, margin: int = 2) -> bool:
    x1, y1, x2, y2 = bbox
    return bool(
        x1 <= margin or y1 <= margin or x2 >= width - 1 - margin or y2 >= height - 1 - margin
    )


def make_detector(
    cfg: dict | None, frame_size: tuple[int, int], roi_mask: np.ndarray | None = None
):
    """Build a detector from a site-config ``detector`` section.

    ``{"backend": "bgsub", ...}`` (default) or ``{"backend": "yolo", "model": "yolov8n.pt"}``.
    """
    cfg = dict(cfg or {})
    backend = cfg.pop("backend", "bgsub")
    if backend == "bgsub":
        from .bgsub import BackgroundSubtractionDetector, BGSubConfig

        return BackgroundSubtractionDetector(BGSubConfig(**cfg), frame_size, roi_mask)
    if backend == "yolo":
        from .yolo import YoloDetector

        return YoloDetector(frame_size=frame_size, roi_mask=roi_mask, **cfg)
    raise ValueError(f"unknown detector backend {backend!r}")
