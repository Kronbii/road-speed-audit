import numpy as np
import pytest

from roadspeed.speed import (
    Lane,
    RoadAxis,
    SpeedConfig,
    estimate_speed,
    estimate_speeds,
    fit_slope,
    split_segments,
)
from roadspeed.tracking import Observation, Track


def test_fit_slope_methods_exact():
    t = np.linspace(0, 2, 30)
    s = 3.0 + 12.5 * t
    for m in ("ols", "huber", "theilsen"):
        b, _, sc = fit_slope(t, s, m)
        assert b == pytest.approx(12.5, abs=1e-9)
        assert sc < 1e-9


def test_robust_fit_ignores_outliers(rng):
    t = np.linspace(0, 2, 60)
    s = 12.5 * t + rng.normal(0, 0.05, t.size)
    s[[5, 17, 40]] += 4.0
    b_ols, _, _ = fit_slope(t, s, "ols")
    b_h, _, _ = fit_slope(t, s, "huber")
    b_ts, _, _ = fit_slope(t, s, "theilsen")
    assert abs(b_h - 12.5) < 0.1 and abs(b_ts - 12.5) < 0.1
    assert abs(b_ols - 12.5) > abs(b_h - 12.5)


def test_batched_fit_matches_single(rng):
    t = np.linspace(0, 1, 25)
    S = 10 * t + rng.normal(0, 0.1, (5, t.size))
    b, _, _ = fit_slope(t, S, "huber")
    for i in range(5):
        assert b[i] == pytest.approx(fit_slope(t, S[i], "huber")[0], rel=1e-12)


def _track_from_point(camera, speed_mps, direction=1, fps=30.0, x=1.75, y0=10.0, n=None):
    """Track whose box bottom-centre is the image of a ground point moving at constant speed."""
    n = n or int(round(fps * 60 / speed_mps))
    obs = []
    for k in range(n):
        t = k / fps
        y = y0 + direction * speed_mps * t if direction > 0 else 60 - speed_mps * t
        if not 5 < y < 70:
            continue
        P = np.array([[x, y, 0.0], [x, y, 1.4]])
        uv, _ = camera.project(P)
        u, v = uv[0]
        box = np.array([u - 30, uv[1, 1], u + 30, v])
        obs.append(Observation(k, t, box, 1.0, False))
    return Track(track_id=1, kf=None, observations=obs)


def test_estimate_speed_exact_projection(camera):
    H = camera.image_to_ground_homography()
    cfg = SpeedConfig(zone_start_m=20, zone_end_m=50)
    lanes = [Lane("a", 1.75, 3.5, "away"), Lane("t", -1.75, 3.5, "towards")]
    for v_kmh, d in ((30.0, 1), (90.0, 1), (60.0, -1)):
        tr = _track_from_point(camera, v_kmh / 3.6, d, x=1.75 if d > 0 else -1.75)
        est = estimate_speed(tr, H, RoadAxis(), cfg, lanes)
        assert est.speed_kmh == pytest.approx(v_kmh, abs=1e-6)
        assert est.direction == ("away" if d > 0 else "towards")
        assert est.lane == ("a" if d > 0 else "t")
        assert est.status == "valid", est.flags


def test_partial_track_rejected(camera):
    H = camera.image_to_ground_homography()
    tr = _track_from_point(camera, 20.0, 1)
    tr.observations = [o for o in tr.observations if o.t < 1.2]  # leaves the view early
    est = estimate_speed(tr, H, RoadAxis(), SpeedConfig(zone_start_m=20, zone_end_m=50))
    assert "partial_zone" in est.flags and est.status == "rejected"


def test_split_segments_reversal_and_jump():
    t = np.arange(60) / 30.0
    s = np.concatenate([20 + 15 * t[:30], 60 - 20 * (t[30:] - t[30])])
    segs = split_segments(t, s)
    assert len(segs) == 2 and segs[0][-1] in (28, 29, 30)
    s2 = np.concatenate([20 + 15 * t[:30], 45 + 15 * (t[30:] - t[30])])  # 10 m jump
    assert len(split_segments(t, s2)) == 2
    s3 = 20 + 15 * t + np.random.default_rng(0).normal(0, 0.05, t.size)
    assert len(split_segments(t, s3)) == 1


def test_identity_transfer_gives_two_records(camera):
    H = camera.image_to_ground_homography()
    a = _track_from_point(camera, 20.0, 1, x=1.75)
    b = _track_from_point(camera, 25.0, -1, x=-1.75)
    t_end = a.observations[-1].t
    for o in b.observations:
        o.t += t_end + 1 / 30
        o.frame += a.observations[-1].frame + 1
    tr = Track(track_id=7, kf=None, observations=a.observations + b.observations)
    ests = estimate_speeds(tr, H, RoadAxis(), SpeedConfig(zone_start_m=20, zone_end_m=50))
    speeds = sorted(e.speed_kmh for e in ests if e.status != "rejected")
    assert speeds == pytest.approx([72.0, 90.0], abs=1e-6)


def test_stitching_joins_pieces_of_one_vehicle(camera):
    from roadspeed.pipeline import finalize
    from roadspeed.stitch import stitch_tracks
    from test_calibration import _site

    full = _track_from_point(camera, 20.0, 1, x=1.75)
    obs = full.observations
    cut = len(obs) // 2
    a = Track(track_id=1, kf=None, observations=obs[: cut - 3])  # 3 frames missing
    b = Track(track_id=2, kf=None, observations=obs[cut:])
    other = _track_from_point(camera, 25.0, -1, x=-1.75)
    other.track_id = 3
    H = camera.image_to_ground_homography()
    out, joins = stitch_tracks([a, b, other], H, RoadAxis(), (20.0, 50.0))
    assert joins == [(1, 2)] and len(out) == 2
    # end to end through finalize: one valid record at the true speed per vehicle
    site = _site(camera)
    site.uncertainty = {"n_samples": 20}
    res = finalize([a, b, other], site, 30.0, 300, 0.0, 10.0)
    valid = sorted(r.speed_kmh for r in res.records if r.status == "valid")
    assert valid == pytest.approx([72.0, 90.0], abs=1e-6)


def test_stitching_refuses_inconsistent_pieces(camera):
    from roadspeed.stitch import stitch_tracks

    a = _track_from_point(camera, 20.0, 1, x=1.75)
    a.observations = [o for o in a.observations if o.t < 1.2]
    b = _track_from_point(camera, 30.0, 1, x=1.75)  # different speed
    b.track_id = 2
    b.observations = [o for o in b.observations if 1.3 < o.t]
    out, joins = stitch_tracks(
        [a, b], camera.image_to_ground_homography(), RoadAxis(), (20.0, 50.0)
    )
    assert joins == []
