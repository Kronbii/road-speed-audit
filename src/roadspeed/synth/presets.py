"""Named scenario presets (all simulated)."""

from __future__ import annotations

from .scene import CameraSpec, LightingSpec, RoadSpec, ScenarioConfig, TrafficSpec, ZoneSpec


def baseline(seed: int = 0, duration_s: float = 120.0) -> ScenarioConfig:
    """Two-lane two-way road, camera on a 6 m roadside pole, 720p at 30 fps."""
    return ScenarioConfig(
        seed=seed,
        duration_s=duration_s,
        road=RoadSpec(lanes_per_direction=1, lane_width_m=3.5),
        camera=CameraSpec(height_m=6.0, hfov_deg=60.0, width=1280, height=720, fps=30.0),
        traffic=TrafficSpec(flow_veh_h_per_lane=300.0, speed_dist="uniform"),
        lighting=LightingSpec(),
        zone=ZoneSpec(20.0, 50.0),
    )


def four_lane(seed: int = 0, duration_s: float = 120.0) -> ScenarioConfig:
    """Four-lane two-way road with heavier traffic (more occlusion)."""
    cfg = baseline(seed, duration_s)
    cfg.road.lanes_per_direction = 2
    cfg.traffic.flow_veh_h_per_lane = 450.0
    return cfg


def small(seed: int = 0, duration_s: float = 10.0) -> ScenarioConfig:
    """Low-resolution short clip for tests and smoke runs."""
    cfg = baseline(seed, duration_s)
    cfg.camera.width, cfg.camera.height = 640, 360
    cfg.traffic.flow_veh_h_per_lane = 600.0
    return cfg


PRESETS = {"baseline": baseline, "four_lane": four_lane, "small": small}
