import numpy as np
import pytest

from roadspeed.geometry import fit_homography
from roadspeed.speed import RoadAxis, SpeedConfig, estimate_speed
from roadspeed.uncertainty import (
    UncertaintyConfig,
    calibration_draws,
    effective_click_sigma,
    propagate,
)
from test_speed import _track_from_point


def _cal(camera):
    g = np.array([[x, y] for y in (20.0, 35.0, 50.0) for x in (-3.5, 0.0, 3.5)])
    uv, _ = camera.project(np.column_stack([g, np.zeros(len(g))]))
    return uv, g


def test_draws_are_reproducible(camera):
    uv, g = _cal(camera)
    H = fit_homography(uv, g)
    cfg = UncertaintyConfig(n_samples=20, click_sigma_px=1.0, ground_sigma_m=0.02, seed=3)
    a, _, _ = calibration_draws(uv, g, cfg, (1280, 720), H)
    b, _, _ = calibration_draws(uv, g, cfg, (1280, 720), H)
    assert np.array_equal(a, b)


def test_zero_noise_gives_zero_calibration_uncertainty(camera):
    uv, g = _cal(camera)
    H = fit_homography(uv, g)
    cfg = UncertaintyConfig(
        n_samples=30, click_sigma_px=0.0, ground_sigma_m=0.0, use_reprojection_residuals=False
    )
    Hs, _, _ = calibration_draws(uv, g, cfg, (1280, 720), H)
    est = estimate_speed(_track_from_point(camera, 20.0), H, RoadAxis(), SpeedConfig())
    res, scale = propagate([est], H, RoadAxis(), "huber", cfg, Hs)
    assert res[0]["u_cal_kmh"] < 1e-6
    assert np.allclose(scale, 1.0)


def test_uncertainty_grows_with_click_noise(camera):
    uv, g = _cal(camera)
    H = fit_homography(uv, g)
    est = estimate_speed(_track_from_point(camera, 20.0), H, RoadAxis(), SpeedConfig())
    us = []
    for sig in (0.5, 2.0):
        cfg = UncertaintyConfig(
            n_samples=200, click_sigma_px=sig, ground_sigma_m=0.0, use_reprojection_residuals=False
        )
        Hs, _, _ = calibration_draws(uv, g, cfg, (1280, 720), H)
        res, _ = propagate([est], H, RoadAxis(), "huber", cfg, Hs)
        us.append(res[0]["u_cal_kmh"])
    assert us[1] == pytest.approx(4 * us[0], rel=0.35)


def test_monte_carlo_matches_repeated_calibrations(camera):
    """The MC spread predicts the spread of speeds over independently noisy calibrations."""
    uv, g = _cal(camera)
    track = _track_from_point(camera, 25.0)
    sig = 1.0
    rng = np.random.default_rng(5)
    speeds = []
    for _ in range(150):
        Hn = fit_homography(uv + rng.normal(0, sig, uv.shape), g)
        speeds.append(estimate_speed(track, Hn, RoadAxis(), SpeedConfig()).speed_kmh)
    H = fit_homography(uv, g)
    cfg = UncertaintyConfig(
        n_samples=300, click_sigma_px=sig, ground_sigma_m=0.0, use_reprojection_residuals=False
    )
    Hs, _, _ = calibration_draws(uv, g, cfg, (1280, 720), H)
    est = estimate_speed(track, H, RoadAxis(), SpeedConfig())
    res, _ = propagate([est], H, RoadAxis(), "huber", cfg, Hs)
    assert res[0]["u_cal_kmh"] == pytest.approx(np.std(speeds), rel=0.25)


def test_effective_click_sigma_from_residuals(camera, rng):
    uv, g = _cal(camera)
    noisy = uv + rng.normal(0, 2.0, uv.shape)
    H = fit_homography(noisy, g)
    s = effective_click_sigma(UncertaintyConfig(click_sigma_px=0.1), noisy, g, H)
    assert 0.8 < s < 4.0
