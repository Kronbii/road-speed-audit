"""Scenario description and traffic generation for the synthetic generator.

Everything here is simulation. A scenario fixes the road layout, a calibrated
pinhole camera, the traffic (arrival process, speeds, vehicle mix), lighting and
sensor effects, and a random seed; the same scenario always produces the same
frames and the same ground truth.
"""

from __future__ import annotations

import copy
import json
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass

import numpy as np

from ..geometry import PinholeCamera
from .vehicles import DEFAULT_MIX, VehicleModel, build_vehicle


@dataclass
class RoadSpec:
    lanes_per_direction: int = 1
    lane_width_m: float = 3.5
    shoulder_m: float = 1.0
    dash_length_m: float = 3.0
    dash_gap_m: float = 9.0
    line_width_m: float = 0.15
    two_way: bool = True
    visible_far_m: float = 400.0  # vehicles are simulated up to this distance
    visible_near_m: float = -15.0  # ... and from this distance (behind the camera)

    @property
    def n_lanes(self) -> int:
        return self.lanes_per_direction * (2 if self.two_way else 1)

    @property
    def width_m(self) -> float:
        return self.n_lanes * self.lane_width_m

    def lane_centers(self) -> np.ndarray:
        w = self.width_m
        return -w / 2 + (np.arange(self.n_lanes) + 0.5) * self.lane_width_m

    def lane_directions(self) -> np.ndarray:
        """+1 for traffic moving away from the camera (+Y), -1 towards it.

        Right-hand traffic: lanes with X < 0 carry traffic towards the camera when the
        camera stands on the right-hand roadside looking along +Y.
        """
        if not self.two_way:
            return np.ones(self.n_lanes, dtype=int)
        n = self.lanes_per_direction
        return np.array([-1] * n + [1] * n, dtype=int)


@dataclass
class CameraSpec:
    height_m: float = 6.0
    offset_m: float = 1.5  # distance of the pole from the right road edge (outwards)
    hfov_deg: float = 60.0
    width: int = 1280
    height: int = 720
    aim_distance_m: float | None = None  # along-road aim point, None = zone centre
    aim_x_m: float = 0.0
    tilt_offset_deg: float = 0.0
    yaw_offset_deg: float = 0.0
    roll_deg: float = 0.0
    fps: float = 30.0
    exposure_s: float = 0.0  # > 0 renders motion blur with `blur_subframes` samples
    blur_subframes: int = 4
    noise_sigma: float = 2.0  # additive Gaussian noise, 8-bit DN
    jpeg_quality: int | None = 90  # None = no compression artefacts


@dataclass
class TrafficSpec:
    flow_veh_h_per_lane: float = 300.0
    speed_dist: str = "uniform"  # "uniform" or "normal"
    speed_min_kmh: float = 20.0
    speed_max_kmh: float = 120.0
    speed_mean_kmh: float = 60.0
    speed_sd_kmh: float = 12.0
    mix: dict = field(default_factory=lambda: dict(DEFAULT_MIX))
    min_headway_s: float = 1.2
    lateral_sd_m: float = 0.25


@dataclass
class LightingSpec:
    gain: float = 1.0
    drift_amplitude: float = 0.0  # fractional sinusoidal gain drift
    drift_period_s: float = 40.0
    jumps: list = field(default_factory=list)  # [[time_s, multiplicative factor], ...]
    sun_elevation_deg: float = 45.0
    sun_azimuth_deg: float = 35.0  # from +Y towards +X
    shadows: bool = True
    shadow_strength: float = 0.45


@dataclass
class ZoneSpec:
    start_m: float = 20.0
    end_m: float = 50.0

    @property
    def center_m(self) -> float:
        return 0.5 * (self.start_m + self.end_m)


@dataclass
class ScenarioConfig:
    seed: int = 0
    duration_s: float = 60.0
    warmup_s: float = 3.0  # ground-truth vehicles crossing the zone before this are ignored
    road: RoadSpec = field(default_factory=RoadSpec)
    camera: CameraSpec = field(default_factory=CameraSpec)
    traffic: TrafficSpec = field(default_factory=TrafficSpec)
    lighting: LightingSpec = field(default_factory=LightingSpec)
    zone: ZoneSpec = field(default_factory=ZoneSpec)
    occluders: list = field(default_factory=list)  # [{"x":, "y":, "w":, "d":, "h":}, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ScenarioConfig:
        return _from_dict(cls, d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def replace(self, **updates) -> ScenarioConfig:
        """Copy with dotted-path updates, e.g. ``replace(**{"camera.height_m": 8})``."""
        new = copy.deepcopy(self)
        for key, value in updates.items():
            obj = new
            parts = key.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            if not hasattr(obj, parts[-1]):
                raise AttributeError(f"unknown scenario field {key!r}")
            setattr(obj, parts[-1], value)
        return new


def _from_dict(cls, d):
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        t = hints.get(f.name)
        if isinstance(t, type) and is_dataclass(t) and isinstance(v, dict):
            v = _from_dict(t, v)
        kwargs[f.name] = v
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Camera placement
# ---------------------------------------------------------------------------


def camera_position(cfg: ScenarioConfig) -> np.ndarray:
    x = cfg.road.width_m / 2 + cfg.road.shoulder_m + cfg.camera.offset_m
    return np.array([x, 0.0, cfg.camera.height_m])


def build_camera(cfg: ScenarioConfig) -> PinholeCamera:
    """Pinhole camera on a roadside pole, aimed at the measurement zone."""
    cam = cfg.camera
    C = camera_position(cfg)
    aim_y = cfg.zone.center_m if cam.aim_distance_m is None else cam.aim_distance_m
    target = np.array([cam.aim_x_m, aim_y, 0.0])
    d = target - C
    yaw = np.degrees(np.arctan2(d[0], d[1])) + cam.yaw_offset_deg
    tilt = np.degrees(np.arctan2(-d[2], np.hypot(d[0], d[1]))) + cam.tilt_offset_deg
    f = PinholeCamera.focal_from_hfov(cam.width, cam.hfov_deg)
    return PinholeCamera.from_pose(f, cam.width, cam.height, C, yaw, tilt, cam.roll_deg)


# ---------------------------------------------------------------------------
# Traffic
# ---------------------------------------------------------------------------


@dataclass
class SimVehicle:
    """A simulated vehicle moving at constant speed along the centre of its lane."""

    vid: int
    lane: int
    direction: int  # +1 away from camera (+Y), -1 towards it
    x_m: float  # lateral position of the footprint centre
    t_center: float  # time the footprint centre crosses the zone centre
    speed_mps: float
    y_center: float  # along-road position of the zone centre
    model: VehicleModel

    def y_at(self, t):
        """Along-road position of the footprint centre (metres, world Y)."""
        return self.y_center + self.direction * self.speed_mps * (np.asarray(t, float) - self.t_center)

    def time_at_y(self, y: float) -> float:
        """Time at which the footprint centre is at along-road position ``y``."""
        return self.t_center + (y - self.y_center) * self.direction / self.speed_mps


def _sample_speed(tr: TrafficSpec, rng: np.random.Generator) -> float:
    if tr.speed_dist == "uniform":
        v = rng.uniform(tr.speed_min_kmh, tr.speed_max_kmh)
    elif tr.speed_dist == "normal":
        while True:
            v = rng.normal(tr.speed_mean_kmh, tr.speed_sd_kmh)
            if tr.speed_min_kmh <= v <= tr.speed_max_kmh:
                break
    else:
        raise ValueError(f"unknown speed distribution {tr.speed_dist!r}")
    return v / 3.6


def generate_traffic(cfg: ScenarioConfig) -> list[SimVehicle]:
    """Poisson arrivals per lane with a minimum time headway and no overtaking in-lane.

    Vehicles are generated for zone-centre crossing times from ``-pre`` to
    ``duration + post`` so the scene is already in steady state at t = 0.
    """
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 1]))
    road, tr, zone = cfg.road, cfg.traffic, cfg.zone
    yc = zone.center_m
    centers = road.lane_centers()
    dirs = road.lane_directions()
    classes = list(tr.mix.keys())
    probs = np.array([tr.mix[c] for c in classes], dtype=float)
    probs /= probs.sum()
    rate = tr.flow_veh_h_per_lane / 3600.0
    pre = (road.visible_far_m - road.visible_near_m) / (tr.speed_min_kmh / 3.6)
    t_lo, t_hi = -min(pre, 60.0), cfg.duration_s + min(pre, 60.0)
    vehicles: list[SimVehicle] = []
    vid = 0
    for lane in range(road.n_lanes):
        d = int(dirs[lane])
        # travel distance before / after the zone centre within the simulated range
        if d > 0:
            before, after = yc - road.visible_near_m, road.visible_far_m - yc
        else:
            before, after = road.visible_far_m - yc, yc - road.visible_near_m
        t = t_lo
        prev = None
        while True:
            t += rng.exponential(1.0 / rate) if rate > 0 else np.inf
            if t > t_hi:
                break
            v = _sample_speed(tr, rng)
            cls = classes[rng.choice(len(classes), p=probs)]
            model = build_vehicle(cls, rng)
            lat = float(np.clip(rng.normal(0, tr.lateral_sd_m), -0.6, 0.6))
            if prev is not None:
                # follower must stay >= headway + its length behind the leader at every
                # position in the simulated range (constant-speed check at both ends)
                need = tr.min_headway_s + (prev.model.length + model.length) / 2 / v
                worst = max(p / prev.speed_mps - p / v for p in (-before, after))
                t = max(t, prev.t_center + need + max(worst, 0.0))
                if t > t_hi:
                    break
            sv = SimVehicle(
                vid=vid,
                lane=lane,
                direction=d,
                x_m=float(centers[lane] + lat),
                t_center=float(t),
                speed_mps=float(v),
                y_center=float(yc),
                model=model,
            )
            vehicles.append(sv)
            prev = sv
            vid += 1
    vehicles.sort(key=lambda s: (s.t_center, s.lane))
    for i, s in enumerate(vehicles):
        s.vid = i
    return vehicles
