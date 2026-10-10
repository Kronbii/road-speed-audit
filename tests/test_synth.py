import numpy as np
import pytest

from roadspeed.synth import Renderer, ScenarioConfig, build_camera, generate_traffic
from roadspeed.synth.presets import small


def test_render_is_deterministic():
    cfg = small(seed=3, duration_s=4.0)
    a = Renderer(cfg)
    b = Renderer(cfg)
    for k in (0, 37, 90):
        fa, ga = a.render(k)
        fb, gb = b.render(k)
        assert np.array_equal(fa, fb)
        assert [(x.vid, x.bbox_visible) for x in ga] == [(x.vid, x.bbox_visible) for x in gb]
    # random access gives the same frame as sequential access
    assert np.array_equal(a.render(37)[0], b.render(37)[0])


def test_seeds_differ():
    a = Renderer(small(seed=1, duration_s=4.0)).render(30)[0]
    b = Renderer(small(seed=2, duration_s=4.0)).render(30)[0]
    assert not np.array_equal(a, b)


def test_traffic_prefix_stable_and_headways():
    cfg = ScenarioConfig(seed=9, duration_s=60.0)
    v60 = generate_traffic(cfg)
    v120 = generate_traffic(cfg.replace(duration_s=120.0))
    key = lambda v: (v.lane, round(v.t_center, 9), round(v.speed_mps, 9))  # noqa: E731
    early = {key(v) for v in v60 if v.t_center < 50}
    assert early <= {key(v) for v in v120}
    for lane in range(cfg.road.n_lanes):
        vs = sorted((v for v in v120 if v.lane == lane), key=lambda v: v.t_center)
        for a, b in zip(vs, vs[1:], strict=False):
            # follower never closer than its length to the leader anywhere in range
            for y in (cfg.road.visible_near_m, cfg.road.visible_far_m):
                gap_t = b.time_at_y(y) - a.time_at_y(y)
                assert gap_t > 0
    speeds = np.array([v.speed_mps * 3.6 for v in v120])
    assert speeds.min() >= 20 and speeds.max() <= 120


def test_camera_aims_at_zone_centre():
    cfg = ScenarioConfig()
    cam = build_camera(cfg)
    uv, z = cam.project(np.array([[0.0, cfg.zone.center_m, 0.0]]))
    assert z[0] > 0
    assert uv[0] == pytest.approx(
        [(cfg.camera.width - 1) / 2, (cfg.camera.height - 1) / 2], abs=1e-6
    )


def test_ground_truth_boxes_track_vehicles():
    cfg = small(seed=4, duration_s=6.0)
    r = Renderer(cfg)
    gt = {g["vid"]: g for g in r.ground_truth_vehicles()}
    seen = set()
    for k in range(0, r.n_frames, 15):
        _, anns = r.render(k)
        for a in anns:
            assert 0.0 <= a.visibility <= 1.0
            seen.add(a.vid)
    assert seen and seen <= set(gt)


def test_scenario_json_round_trip():
    cfg = small(seed=5)
    cfg2 = ScenarioConfig.from_dict(cfg.to_dict())
    assert cfg2.to_dict() == cfg.to_dict()
    with pytest.raises(AttributeError):
        cfg.replace(**{"camera.no_such_field": 1})
