import numpy as np
import pytest

from roadspeed.geometry import (
    PinholeCamera,
    apply_homography,
    fit_homography,
    fit_homography_dlt,
    homography_jacobian,
    reprojection_report,
)
from roadspeed.speed import project_to_road


def grid(n=3):
    xs = np.array([-3.5, 0.0, 3.5])
    ys = np.linspace(20, 50, n)
    return np.array([[x, y] for y in ys for x in xs])


def test_dlt_exact_four_points():
    src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    H_true = np.array([[1.2, 0.1, 3.0], [-0.05, 0.9, 1.0], [1e-3, 2e-3, 1.0]])
    dst = apply_homography(H_true, src)
    H = fit_homography_dlt(src, dst)
    assert np.allclose(H / H[2, 2], H_true / H_true[2, 2], atol=1e-9)


def test_degenerate_points_rejected():
    src = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float)
    with pytest.raises(ValueError):
        fit_homography_dlt(src, src)
    with pytest.raises(ValueError):
        fit_homography_dlt(src[:3], src[:3])


def test_camera_homography_round_trip(camera):
    g = grid()
    uv, depth = camera.project(np.column_stack([g, np.zeros(len(g))]))
    assert np.all(depth > 0)
    H = fit_homography(uv, g)
    assert np.allclose(apply_homography(H, uv), g, atol=1e-6)
    assert np.allclose(H, camera.image_to_ground_homography(), atol=1e-6 * np.abs(H).max())
    rep = reprojection_report(H, uv, g)
    assert rep.rms_px < 1e-6 and rep.rms_m < 1e-6


def test_backproject_matches_homography(camera):
    uv = np.array([[640.0, 500.0], [300.0, 650.0], [900.0, 400.0]])
    a = camera.backproject_to_plane(uv, 0.0)
    b = apply_homography(camera.image_to_ground_homography(), uv)
    assert np.allclose(a, b, atol=1e-6)


def test_height_correction_equals_plane_intersection(camera):
    uv = np.array([[640.0, 450.0], [500.0, 600.0]])
    z = 0.5
    exact = camera.backproject_to_plane(uv, z)
    corr = project_to_road(camera.image_to_ground_homography(), uv, camera.C, z)
    assert np.allclose(exact, corr, atol=1e-6)


def test_recover_camera_from_homography(camera):
    H = camera.image_to_ground_homography()
    cam2 = PinholeCamera.from_homography(H, 1280, 720, reference_ground_point=(0, 35))
    assert cam2.focal_px == pytest.approx(camera.focal_px, rel=1e-6)
    assert cam2.camera_height == pytest.approx(6.0, abs=1e-6)
    assert cam2.tilt_deg == pytest.approx(9.6, abs=1e-6)
    assert cam2.yaw_deg == pytest.approx(-9.7, abs=1e-6)


def test_recover_camera_noisy(camera, rng):
    g = grid(5)
    uv, _ = camera.project(np.column_stack([g, np.zeros(len(g))]))
    H = fit_homography(uv + rng.normal(0, 0.5, uv.shape), g)
    cam2 = PinholeCamera.from_homography(H, 1280, 720, reference_ground_point=g.mean(axis=0))
    assert abs(cam2.camera_height - 6.0) < 0.3
    assert abs(cam2.focal_px / camera.focal_px - 1) < 0.05


def test_batched_homography(camera):
    H = camera.image_to_ground_homography()
    Hs = np.stack([H, H * 2.0])
    uv = np.array([[640.0, 500.0], [300.0, 650.0]])
    out = apply_homography(Hs, uv)
    assert out.shape == (2, 2, 2)
    assert np.allclose(out[0], out[1])


def test_jacobian_numeric(camera):
    H = camera.image_to_ground_homography()
    uv = np.array([[600.0, 420.0]])
    J = homography_jacobian(H, uv)[0]
    eps = 1e-3
    num = np.column_stack(
        [
            (apply_homography(H, uv + [eps, 0]) - apply_homography(H, uv - [eps, 0]))[0]
            / (2 * eps),
            (apply_homography(H, uv + [0, eps]) - apply_homography(H, uv - [0, eps]))[0]
            / (2 * eps),
        ]
    )
    assert np.allclose(J, num, rtol=1e-5, atol=1e-8)


def test_horizon_row(camera):
    v = camera.horizon_row()
    # a far point on the road projects just below the horizon
    uv, _ = camera.project(np.array([[0.0, 5000.0, 0.0]]))
    assert uv[0, 1] > v and uv[0, 1] - v < 2.0
