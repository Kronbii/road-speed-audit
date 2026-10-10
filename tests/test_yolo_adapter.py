import sys
import types

import numpy as np
import pytest


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = (
            np.array(xyxy, float),
            np.array(conf, float),
            np.array(cls, float),
        )

    def __len__(self):
        return len(self.conf)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    def __init__(self, path):
        self.path = path

    def predict(self, frame, **kw):
        return [
            _Result(
                _Boxes(
                    [[10, 20, 60, 50], [100, 100, 140, 130], [0, 0, 5, 5]],
                    [0.9, 0.4, 0.8],
                    [2, 7, 0],
                )
            )
        ]


@pytest.fixture
def fake_ultralytics(monkeypatch, tmp_path):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    w = tmp_path / "w.pt"
    w.write_bytes(b"")
    return str(w)


def test_yolo_adapter_maps_vehicle_classes(fake_ultralytics):
    from roadspeed.detect import make_detector

    d = make_detector({"backend": "yolo", "model": fake_ultralytics}, (640, 360))
    dets = d.detect(np.zeros((360, 640, 3), np.uint8))
    assert [x.label for x in dets] == ["car", "truck"]  # class 0 (person) is dropped
    assert dets[0].score == pytest.approx(0.9)


def test_yolo_requires_local_weights(fake_ultralytics):
    from roadspeed.detect.yolo import YoloDetector

    with pytest.raises(FileNotFoundError):
        YoloDetector((640, 360), model="does-not-exist.pt")
