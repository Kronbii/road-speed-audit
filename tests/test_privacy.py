import cv2
import numpy as np

from roadspeed.privacy import anonymize, anonymized_crop


def test_anonymize_removes_detail(rng):
    img = (rng.random((120, 200, 3)) * 255).astype(np.uint8)
    cv2.putText(img, "AB 123456", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    out = anonymize(img)
    lap = lambda x: cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()  # noqa: E731
    assert out.shape == img.shape
    assert lap(out) < 0.01 * lap(img)


def test_crop_size():
    frame = np.zeros((360, 640, 3), np.uint8)
    c = anonymized_crop(frame, (100, 100, 180, 150), out_size=96)
    assert c.shape == (96, 96, 3)
