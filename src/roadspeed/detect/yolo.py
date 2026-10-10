"""Optional YOLO detector backend (``pip install roadspeed[yolo]``).

Uses the ``ultralytics`` package. Offline first: the weights file must already be
on disk unless ``allow_download=True`` is passed explicitly. This backend is not
exercised by the synthetic experiments in this repository (those use background
subtraction); its adapter logic is unit-tested with a stand-in model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Detection, touches_border

# COCO class ids for road vehicles
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YoloDetector:
    def __init__(
        self,
        frame_size: tuple[int, int],
        model: str = "yolov8n.pt",
        conf: float = 0.1,
        device: str | None = None,
        classes: dict | None = None,
        roi_mask: np.ndarray | None = None,
        allow_download: bool = False,
        imgsz: int | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "the YOLO backend needs the optional extra: pip install 'roadspeed[yolo]'"
            ) from e
        if not allow_download and not Path(model).exists():
            raise FileNotFoundError(
                f"weights {model!r} not found locally; download them once yourself or pass "
                "allow_download=True (this tool does not fetch anything by default)"
            )
        self.model = YOLO(model)
        self.width, self.height = frame_size
        self.conf = conf
        self.device = device
        self.classes = dict(
            COCO_VEHICLES if classes is None else {int(k): v for k, v in classes.items()}
        )
        self.roi_mask = roi_mask
        self.imgsz = imgsz

    def prime(self, frames) -> None:  # interface parity with background subtraction
        return None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        kwargs = dict(conf=self.conf, classes=list(self.classes), verbose=False)
        if self.device is not None:
            kwargs["device"] = self.device
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        results = self.model.predict(frame, **kwargs)
        dets: list[Detection] = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = _np(b.xyxy)
            conf = _np(b.conf)
            cls = _np(b.cls).astype(int)
            for box, c, k in zip(xyxy, conf, cls, strict=True):
                if int(k) not in self.classes:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                if self.roi_mask is not None:
                    u, v = int(round(0.5 * (x1 + x2))), int(round(min(y2, self.height - 1)))
                    if not self.roi_mask[
                        min(max(v, 0), self.height - 1), min(max(u, 0), self.width - 1)
                    ]:
                        continue
                bb = np.array([x1, y1, x2, y2])
                dets.append(
                    Detection(
                        bbox=bb,
                        score=float(c),
                        label=self.classes[int(k)],
                        area_px=int((x2 - x1) * (y2 - y1)),
                        touches_border=touches_border(bb, self.width, self.height),
                    )
                )
        return dets


def _np(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)
