import cv2
import numpy as np
import pytest

cv2.setNumThreads(1)


@pytest.fixture
def camera():
    from roadspeed.geometry import PinholeCamera

    f = PinholeCamera.focal_from_hfov(1280, 60.0)
    return PinholeCamera.from_pose(f, 1280, 720, [6.0, 0.0, 6.0], yaw_deg=-9.7, tilt_deg=9.6)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
