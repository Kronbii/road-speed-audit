import numpy as np
import pytest

from roadspeed.calibration import Calibration, SiteConfig, points_from_lane_markings
from roadspeed.geometry import apply_homography, fit_homography


def _site(camera, n=3):
    xs = [-3.5, 0.0, 3.5]
    ys = np.linspace(20, 50, n)
    g = np.array([[x, y] for y in ys for x in xs])
    uv, _ = camera.project(np.column_stack([g, np.zeros(len(g))]))
    cal = Calibration(image_points=uv.tolist(), ground_points=g.tolist(), image_size=[1280, 720])
    lanes = [
        {"name": "t", "center_m": -1.75, "width_m": 3.5, "direction": "towards"},
        {"name": "a", "center_m": 1.75, "width_m": 3.5, "direction": "away"},
    ]
    return SiteConfig(name="x", calibration=cal, lanes=lanes, speed_limit_kmh=50)


def test_site_round_trip(tmp_path, camera):
    site = _site(camera)
    p = tmp_path / "site.json"
    site.save(p)
    s2 = SiteConfig.load(p)
    assert s2.to_dict() == site.to_dict()
    assert np.allclose(s2.calibration.homography(), site.calibration.homography())


def test_unknown_keys_rejected(camera):
    d = _site(camera).to_dict()
    d["zonee"] = {}
    with pytest.raises(ValueError):
        SiteConfig.from_dict(d)


def test_report_and_camera(camera):
    rep = _site(camera).calibration.report()
    assert rep["n_points"] == 9 and rep["rms_px"] < 1e-6
    assert rep["camera"]["height_m"] == pytest.approx(6.0, abs=1e-3)


def test_zone_visibility(camera):
    site = _site(camera)
    vis = site.zone_visibility()
    assert vis["all_inside"]
    site.zone = {"start_m": 2.0, "end_m": 50.0}
    assert not site.zone_visibility()["all_inside"]


def test_roi_mask(camera):
    m = _site(camera).roi_mask()
    assert m is not None and 0.1 < m.mean() < 0.9


def test_lane_marking_helper_recovers_geometry(camera):
    dash, gap = 3.0, 9.0
    lines = []
    for x in (0.0, 3.5):
        pts = []
        for k in range(4):
            for end, off in (("near", 0.0), ("far", dash)):
                y = 20.0 + k * (dash + gap) + off
                uv, _ = camera.project(np.array([[x, y, 0.0]]))
                pts.append({"uv": uv[0].tolist(), "dash": k, "end": end})
        lines.append({"x_m": x, "phase_m": 20.0, "points": pts})
    img, gnd = points_from_lane_markings(lines, dash, gap)
    H = fit_homography(img, gnd)
    uv, _ = camera.project(np.array([[1.75, 33.0, 0.0]]))
    assert np.allclose(apply_homography(H, uv)[0], [1.75, 33.0], atol=1e-6)


def test_lane_marking_shear_does_not_change_along_lane_distance(camera):
    """Wrong dash phase between lines = ground shear; along-lane distances are unchanged."""
    dash, gap = 3.0, 9.0
    lines = []
    for x, true_phase in ((0.0, 20.0), (3.5, 24.0)):
        pts = []
        for k in range(4):
            for end, off in (("near", 0.0), ("far", dash)):
                y = true_phase + k * (dash + gap) + off
                uv, _ = camera.project(np.array([[x, y, 0.0]]))
                pts.append({"uv": uv[0].tolist(), "dash": k, "end": end})
        lines.append({"x_m": x, "phase_m": 0.0, "points": pts})  # phases unknown
    img, gnd = points_from_lane_markings(lines, dash, gap)
    H = fit_homography(img, gnd)
    # a vehicle in the lane between the lines moves from y=25 to y=45 (20 m)
    a, b = camera.project(np.array([[1.75, 25.0, 0.0], [1.75, 45.0, 0.0]]))[0]
    pa, pb = apply_homography(H, np.array([a, b]))
    assert abs(pb[1] - pa[1]) == pytest.approx(20.0, abs=1e-6)
    assert pb[0] == pytest.approx(pa[0], abs=1e-6)
