"""Speed estimation from a track: contact point -> road plane -> robust fit over the zone.

For every observation the bottom-centre of the box (where the vehicle meets the
road) is mapped to the road plane with the calibration homography and expressed
as an along-road coordinate ``s`` and a lateral coordinate ``l``. The speed is
the slope of a robust straight-line fit of ``s`` against time using only the
observations inside the measurement zone; checks on the fit and the track set
quality flags that decide whether a record is valid, needs human review, or is
rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import apply_homography

# Flags that make a speed unusable (rejected) vs. flags that send it to review.
HARD_FLAGS = {"few_points", "partial_zone", "implausible_speed", "no_zone_points"}
SOFT_FLAGS = {
    "high_residual",
    "gap",
    "wide_blob",
    "lane_ambiguous",
    "direction_mismatch",
    "high_uncertainty",
    "border",
    "mostly_merged",
}
# Informational flags: recorded, but they do not change the status on their own.
INFO_FLAGS = {"split"}


@dataclass
class Lane:
    name: str
    center_m: float  # lateral coordinate of the lane centre (metres, + = right of road axis)
    width_m: float = 3.5
    direction: str = "both"  # "away", "towards" or "both"


@dataclass
class RoadAxis:
    """Along-road coordinate system on the ground plane."""

    origin: tuple = (0.0, 0.0)
    direction: tuple = (0.0, 1.0)  # unit vector, + = "away" from the camera

    def __post_init__(self):
        d = np.asarray(self.direction, float)
        d = d / np.linalg.norm(d)
        self.direction = (float(d[0]), float(d[1]))

    def to_sl(self, xy: np.ndarray) -> np.ndarray:
        """Ground ``(..., 2)`` -> ``(..., 2)`` with columns (along s, lateral l)."""
        o = np.asarray(self.origin, float)
        d = np.asarray(self.direction, float)
        n = np.array([d[1], -d[0]])
        p = np.asarray(xy, float) - o
        return np.stack([p @ d, p @ n], axis=-1)


@dataclass
class SpeedConfig:
    zone_start_m: float = 20.0
    zone_end_m: float = 50.0
    fit: str = "huber"  # "huber", "theilsen" or "ols"
    min_points: int = 6
    min_coverage: float = 0.7  # fraction of the zone length spanned by in-zone points
    max_residual_m: float = 0.5  # robust residual scale above which the fit is flagged
    min_speed_kmh: float = 3.0
    max_speed_kmh: float = 200.0
    max_gap_s: float = 0.5
    max_width_m: float = 6.0  # ground width of the blob bottom; wider suggests merged vehicles
    max_lateral_sd_m: float = 0.6
    max_uncertainty_kmh: float = 3.0
    contact: str = "bottom"  # "bottom": bbox bottom-centre; "center": bbox centre
    contact_height_m: float = 0.0  # height of the reference point above the road
    border_margin_px: int = 2
    split_margin_m: float = 15.0  # tracks are checked for identity jumps within zone +- this


@dataclass
class SpeedEstimate:
    track_id: int
    speed_kmh: float | None
    direction: str | None
    lane: str | None
    t_zone_in: float | None
    t_zone_out: float | None
    t_mid: float | None
    n_points: int
    coverage: float
    residual_m: float
    lateral_m: float | None
    lateral_sd_m: float
    width_m: float | None
    max_gap_s: float
    flags: list[str] = field(default_factory=list)
    label: str = "vehicle"
    segment: int = 0
    # data kept for uncertainty propagation and evaluation (not serialised)
    uv: np.ndarray | None = None
    t: np.ndarray | None = None
    zone_idx: np.ndarray | None = None
    frames: np.ndarray | None = None

    @property
    def status(self) -> str:
        if any(f in HARD_FLAGS for f in self.flags):
            return "rejected"
        if any(f not in INFO_FLAGS for f in self.flags):
            return "review"
        return "valid"


# ---------------------------------------------------------------------------
# Robust line fits (vectorised over rows so Monte Carlo can reuse them)
# ---------------------------------------------------------------------------


def fit_slope(
    t: np.ndarray, S: np.ndarray, method: str = "huber", k: float = 1.345, iters: int = 10
):
    """Fit ``S = a + b t`` for each row of ``S``.

    ``t`` is ``(n,)`` and ``S`` is ``(n,)`` or ``(M, n)``. Returns ``(b, a, scale)``
    with the intercept ``a`` defined at ``t = mean(t)`` and ``scale`` a robust
    (MAD-based) residual scale.
    """
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    single = S.ndim == 1
    S2 = S[None, :] if single else S
    tc = t - t.mean()
    if method == "theilsen":
        i, j = np.triu_indices(len(t), 1)
        dt = tc[j] - tc[i]
        ok = np.abs(dt) > 1e-9
        slopes = (S2[:, j[ok]] - S2[:, i[ok]]) / dt[ok]
        b = np.median(slopes, axis=1)
        a = np.median(S2 - b[:, None] * tc, axis=1)
    else:
        w = np.ones_like(S2)
        n_iter = 1 if method == "ols" else iters
        for _ in range(n_iter):
            sw = w.sum(axis=1)
            mt = (w * tc).sum(axis=1) / sw
            ms = (w * S2).sum(axis=1) / sw
            dtc = tc[None, :] - mt[:, None]
            b = (w * dtc * (S2 - ms[:, None])).sum(axis=1) / np.maximum(
                (w * dtc**2).sum(axis=1), 1e-12
            )
            a = ms - b * mt
            if method == "ols":
                break
            r = S2 - a[:, None] - b[:, None] * tc
            sc = 1.4826 * np.median(np.abs(r - np.median(r, axis=1, keepdims=True)), axis=1)
            sc = np.maximum(sc, 1e-4)
            u = np.abs(r) / (k * sc[:, None])
            w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
    r = S2 - a[:, None] - b[:, None] * tc
    scale = 1.4826 * np.median(np.abs(r - np.median(r, axis=1, keepdims=True)), axis=1)
    if single:
        return float(b[0]), float(a[0]), float(scale[0])
    return b, a, scale


# ---------------------------------------------------------------------------
# Projection of contact points
# ---------------------------------------------------------------------------


def contact_points(bboxes: np.ndarray, mode: str = "bottom") -> np.ndarray:
    b = np.asarray(bboxes, float).reshape(-1, 4)
    u = 0.5 * (b[:, 0] + b[:, 2])
    v = b[:, 3] if mode == "bottom" else 0.5 * (b[:, 1] + b[:, 3])
    return np.stack([u, v], axis=1)


def project_to_road(H, uv, camera_center=None, height_m: float = 0.0) -> np.ndarray:
    """Map pixels to road-plane XY, optionally for a reference point at ``height_m``.

    A point at height ``z`` seen through the ground homography lands too far from
    the camera by the factor ``h / (h - z)``; with the camera centre ``(Cx, Cy, h)``
    known (for example recovered with :meth:`PinholeCamera.from_homography`) the
    true foot point is ``C + (P - C) (h - z) / h``. ``H`` and ``camera_center`` may be
    batched (``(M, 3, 3)`` and ``(M, 3)``).
    """
    P = apply_homography(H, uv)
    if height_m == 0.0 or camera_center is None:
        return P
    C = np.asarray(camera_center, float)
    if C.ndim == 1:
        f = (C[2] - height_m) / C[2]
        return C[:2] + (P - C[:2]) * f
    f = (C[:, 2] - height_m) / C[:, 2]
    return C[:, None, :2] + (P - C[:, None, :2]) * f[:, None, None]


def assign_lane(lateral: float, lanes: list[Lane]) -> str | None:
    best, bd = None, np.inf
    for ln in lanes:
        d = abs(lateral - ln.center_m)
        if d <= ln.width_m / 2 and d < bd:
            best, bd = ln, d
    return None if best is None else best.name


# ---------------------------------------------------------------------------
# Per-track estimate
# ---------------------------------------------------------------------------


def split_segments(
    t, s, max_speed_mps: float = 200 / 3.6, min_len: int = 6, win: int = 5, max_gap_s: float = 1.0
) -> list[np.ndarray]:
    """Split a track where its along-road position jumps or reverses direction.

    When two vehicles overlap in the image (typically opposite directions meeting
    far from the camera), the detector sees one blob and the tracker can carry an
    identity from one vehicle to the other. In road coordinates that shows up as a
    jump or a reversal of travel direction; each piece is then measured separately.
    Returns index arrays into ``t`` / ``s``.
    """
    t = np.asarray(t, float)
    s = np.asarray(s, float)
    n = len(t)
    if n < 2:
        return [np.arange(n)]
    dt = np.maximum(np.diff(t), 1e-6)
    ds = np.diff(s)
    cuts = set((np.nonzero(np.abs(ds) > np.maximum(2.0 * max_speed_mps * dt, 4.0))[0] + 1).tolist())
    cuts |= set((np.nonzero(dt > max_gap_s)[0] + 1).tolist())
    if n >= 2 * min_len:
        v = ds / dt
        pad = np.pad(v, win, mode="edge")
        vm = np.array([np.median(pad[i : i + 2 * win + 1]) for i in range(len(v))])
        sg = np.where(vm > 1.0, 1, np.where(vm < -1.0, -1, 0))
        # run-length encode non-zero signs, drop short runs
        runs = []  # (sign, start, end) over step indices
        i = 0
        while i < len(sg):
            j = i
            while j + 1 < len(sg) and sg[j + 1] == sg[i]:
                j += 1
            if sg[i] != 0 and (j - i + 1) >= min_len:
                runs.append((sg[i], i, j))
            i = j + 1
        for (s1, _, e1), (s2, b2, _) in zip(runs, runs[1:], strict=False):
            if s1 != s2:
                cuts.add((e1 + b2) // 2 + 1)
    edges = [0] + sorted(c for c in cuts if 0 < c < n) + [n]
    segs = [np.arange(a, b) for a, b in zip(edges, edges[1:], strict=False) if b - a >= 2]
    return segs or [np.arange(n)]


def estimate_speeds(
    track,
    H: np.ndarray,
    axis: RoadAxis,
    cfg: SpeedConfig,
    lanes: list[Lane] | None = None,
    image_size: tuple[int, int] | None = None,
    camera_center=None,
) -> list[SpeedEstimate]:
    """Speed estimates for one track: one per segment that reaches the zone."""
    obs = [o for o in track.observations if not o.touches_border]
    if len(obs) < 2:
        return [estimate_speed(track, H, axis, cfg, lanes, image_size, camera_center)]
    t = np.array([o.t for o in obs])
    uv = contact_points(np.array([o.bbox for o in obs]), cfg.contact)
    s = axis.to_sl(project_to_road(H, uv, camera_center, cfg.contact_height_m))[:, 0]
    # only the stretch around the zone matters; far-away jitter must not cause splits
    z0, z1 = min(cfg.zone_start_m, cfg.zone_end_m), max(cfg.zone_start_m, cfg.zone_end_m)
    near = np.nonzero((s >= z0 - cfg.split_margin_m) & (s <= z1 + cfg.split_margin_m))[0]
    if len(near) < 2:
        return [estimate_speed(track, H, axis, cfg, lanes, image_size, camera_center)]
    segs = [near[idx] for idx in split_segments(t[near], s[near], cfg.max_speed_kmh / 3.6)]
    if len(segs) == 1:
        return [estimate_speed(track, H, axis, cfg, lanes, image_size, camera_center)]
    out = []
    for k, idx in enumerate(segs):
        sub = _SubTrack(track, [obs[i] for i in idx], k)
        e = estimate_speed(sub, H, axis, cfg, lanes, image_size, camera_center)
        e.flags.append("split")
        out.append(e)
    return out


class _SubTrack:
    def __init__(self, track, observations, segment):
        self.track_id = track.track_id
        self.observations = observations
        self.segment = segment
        self.label = track.label


def estimate_speed(
    track,
    H: np.ndarray,
    axis: RoadAxis,
    cfg: SpeedConfig,
    lanes: list[Lane] | None = None,
    image_size: tuple[int, int] | None = None,
    camera_center=None,
) -> SpeedEstimate:
    """Speed for one tracked vehicle (see module docstring)."""
    lanes = lanes or []
    obs = [o for o in track.observations if not o.touches_border]
    est = SpeedEstimate(
        track_id=track.track_id,
        speed_kmh=None,
        direction=None,
        lane=None,
        t_zone_in=None,
        t_zone_out=None,
        t_mid=None,
        n_points=0,
        coverage=0.0,
        residual_m=float("nan"),
        lateral_m=None,
        lateral_sd_m=float("nan"),
        width_m=None,
        max_gap_s=float("nan"),
        label=track.label,
        segment=getattr(track, "segment", 0),
    )
    if len(obs) < 2:
        est.flags.append("no_zone_points")
        return est
    t = np.array([o.t for o in obs])
    bb = np.array([o.bbox for o in obs])
    uv = contact_points(bb, cfg.contact)
    xy = project_to_road(H, uv, camera_center, cfg.contact_height_m)
    sl = axis.to_sl(xy)
    s, lat = sl[:, 0], sl[:, 1]
    z0, z1 = min(cfg.zone_start_m, cfg.zone_end_m), max(cfg.zone_start_m, cfg.zone_end_m)
    inz = np.nonzero((s >= z0) & (s <= z1) & np.isfinite(s))[0]
    est.n_points = int(len(inz))
    if len(inz) < 2:
        est.flags.append("no_zone_points")
        return est
    tz, sz = t[inz], s[inz]
    est.coverage = float((sz.max() - sz.min()) / (z1 - z0))
    est.max_gap_s = float(np.max(np.diff(np.sort(tz)))) if len(tz) > 1 else float("nan")
    b, a, scale = fit_slope(tz, sz, cfg.fit)
    est.residual_m = float(scale)
    v = abs(b) * 3.6
    est.speed_kmh = float(v)
    est.direction = "away" if b > 0 else "towards"
    if abs(b) > 1e-9:
        tm = tz.mean()
        t_at = lambda sv: tm + (sv - a) / b  # noqa: E731
        ta, tb_ = t_at(z0), t_at(z1)
        est.t_zone_in, est.t_zone_out = float(min(ta, tb_)), float(max(ta, tb_))
        est.t_mid = float(t_at(0.5 * (z0 + z1)))
    est.lateral_m = float(np.median(lat[inz]))
    est.lateral_sd_m = float(np.std(lat[inz]))
    est.lane = assign_lane(est.lateral_m, lanes) if lanes else None
    # ground width of the blob's bottom edge (falls back to the box width)
    bw = np.array(
        [
            o.bottom_width_px
            if getattr(o, "bottom_width_px", None) is not None
            else o.bbox[2] - o.bbox[0]
            for o in (obs[i] for i in inz)
        ]
    )
    uc = 0.5 * (bb[inz, 0] + bb[inz, 2])
    left = np.stack([uc - bw / 2, bb[inz, 3]], axis=1)
    right = np.stack([uc + bw / 2, bb[inz, 3]], axis=1)
    wl = np.linalg.norm(apply_homography(H, right) - apply_homography(H, left), axis=1)
    est.width_m = float(np.median(wl))
    est.uv, est.t, est.zone_idx = uv, t, inz
    est.frames = np.array([o.frame for o in obs])

    # quality flags
    if est.n_points < cfg.min_points:
        est.flags.append("few_points")
    if est.coverage < cfg.min_coverage:
        est.flags.append("partial_zone")
    if not (cfg.min_speed_kmh <= v <= cfg.max_speed_kmh):
        est.flags.append("implausible_speed")
    if scale > cfg.max_residual_m:
        est.flags.append("high_residual")
    if est.max_gap_s > cfg.max_gap_s:
        est.flags.append("gap")
    if est.width_m > cfg.max_width_m:
        est.flags.append("wide_blob")
    if est.lateral_sd_m > cfg.max_lateral_sd_m or (lanes and est.lane is None):
        est.flags.append("lane_ambiguous")
    if lanes and est.lane is not None:
        ln = next(x for x in lanes if x.name == est.lane)
        if ln.direction in ("away", "towards") and ln.direction != est.direction:
            est.flags.append("direction_mismatch")
    merged = np.array([getattr(obs[i], "merged", False) for i in inz])
    if merged.mean() > 0.5:
        est.flags.append("mostly_merged")
    n_border = len(track.observations) - len(obs)
    if n_border and image_size is not None:
        # a border-touching observation inside the zone means the zone is not fully in view
        bb_all = np.array([o.bbox for o in track.observations if o.touches_border])
        xy_b = project_to_road(
            H, contact_points(bb_all, cfg.contact), camera_center, cfg.contact_height_m
        )
        sb = axis.to_sl(xy_b)[:, 0]
        if np.any((sb >= z0) & (sb <= z1)):
            est.flags.append("border")
    return est
