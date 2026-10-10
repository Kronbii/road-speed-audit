"""Tracklet stitching in road coordinates.

When two vehicles overlap in the image the tracker sometimes loses one of them and
starts a new track a few frames later, leaving two partial tracks that each cover
only part of the measurement zone. In road coordinates the pieces of one vehicle
are easy to recognise: same lane, same direction, nearly the same speed, and the
second piece starts where the first one's straight-line motion predicts. This
module joins such pieces before speed estimation. It only joins tracks in time
order (the second starts after the first ends, up to a short overlap) and never
across lanes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .speed import RoadAxis, contact_points, fit_slope, project_to_road
from .tracking import Track


@dataclass
class StitchConfig:
    enabled: bool = True
    max_gap_s: float = 1.5  # time between the end of one piece and the start of the next
    max_overlap_s: float = 0.25
    tol_m: float = 1.5  # base tolerance on the along-road position at the junction
    tol_per_s: float = 1.0  # extra tolerance per second of gap (m/s)
    max_lateral_m: float = 1.2
    max_speed_ratio: float = 0.2  # |v1 - v2| / mean(v) at most this
    min_points: int = 4
    end_points: int = 15  # points at each end used for the local line fits
    margin_m: float = 15.0  # only observations within zone +- margin are considered


def _road_coords(track: Track, H, axis: RoadAxis, zone, margin):
    obs = [o for o in track.observations if not o.touches_border]
    if not obs:
        return None
    t = np.array([o.t for o in obs])
    sl = axis.to_sl(project_to_road(H, contact_points(np.array([o.bbox for o in obs]))))
    z0, z1 = min(zone), max(zone)
    m = (sl[:, 0] >= z0 - margin) & (sl[:, 0] <= z1 + margin)
    if m.sum() < 2:
        return None
    return t[m], sl[m, 0], sl[m, 1]


def stitch_tracks(tracks: list[Track], H, axis: RoadAxis, zone, cfg: StitchConfig | None = None):
    """Join partial tracks of the same vehicle. Returns ``(tracks, joins)``.

    ``joins`` lists ``(kept_id, absorbed_id)``. The joined track keeps the first
    piece's id; its observations keep their frame order.
    """
    cfg = cfg or StitchConfig()
    if not cfg.enabled or len(tracks) < 2:
        return list(tracks), []
    info = {}
    for tr in tracks:
        rc = _road_coords(tr, H, axis, zone, cfg.margin_m)
        if rc is None or len(rc[0]) < cfg.min_points:
            continue
        t, s, lat = rc
        k = min(cfg.end_points, len(t))
        b_end, a_end, _ = fit_slope(t[-k:], s[-k:], "huber")
        b_start, a_start, _ = fit_slope(t[:k], s[:k], "huber")
        info[tr.track_id] = {
            "t0": float(t[0]),
            "t1": float(t[-1]),
            "end_line": (b_end, a_end, float(t[-k:].mean())),
            "start_line": (b_start, a_start, float(t[:k].mean())),
            "lat_end": float(np.median(lat[-k:])),
            "lat_start": float(np.median(lat[:k])),
        }
    cands = []
    ids = sorted(info)
    for i in ids:
        A = info[i]
        bA, aA, tA = A["end_line"]
        for j in ids:
            if i == j:
                continue
            B = info[j]
            gap = B["t0"] - A["t1"]
            if gap < -cfg.max_overlap_s or gap > cfg.max_gap_s:
                continue
            bB, aB, tB = B["start_line"]
            if abs(bA) < 1.0 or abs(bB) < 1.0 or np.sign(bA) != np.sign(bB):
                continue
            if abs(bA - bB) > cfg.max_speed_ratio * 0.5 * (abs(bA) + abs(bB)):
                continue
            if abs(A["lat_end"] - B["lat_start"]) > cfg.max_lateral_m:
                continue
            tm = 0.5 * (A["t1"] + B["t0"])
            sA = aA + bA * (tm - tA)
            sB = aB + bB * (tm - tB)
            tol = cfg.tol_m + cfg.tol_per_s * max(gap, 0.0)
            d = abs(sA - sB)
            if d <= tol:
                cands.append((d / tol, i, j))
    cands.sort()
    nxt, prv = {}, {}
    for _, i, j in cands:
        if i in nxt or j in prv:
            continue
        # avoid cycles
        k = i
        while k in prv:
            k = prv[k]
        if k == j:
            continue
        nxt[i], prv[j] = j, i
    if not nxt:
        return list(tracks), []
    by_id = {t.track_id: t for t in tracks}
    out, joins, used = [], [], set()
    for tr in sorted(tracks, key=lambda t: t.track_id):
        if tr.track_id in used or tr.track_id in prv:
            continue
        chain = [tr.track_id]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        if len(chain) == 1:
            out.append(tr)
            continue
        obs = {}
        for tid in chain:
            for o in by_id[tid].observations:
                obs.setdefault(o.frame, o)  # earlier piece wins on overlapping frames
            used.add(tid)
        merged = Track(
            track_id=chain[0],
            kf=by_id[chain[-1]].kf,
            state=by_id[chain[-1]].state,
            hits=sum(by_id[t].hits for t in chain),
            last_frame=max(by_id[t].last_frame for t in chain),
            observations=[obs[f] for f in sorted(obs)],
        )
        merged.stitched_from = chain  # type: ignore[attr-defined]
        joins += [(chain[0], t) for t in chain[1:]]
        out.append(merged)
    out.sort(key=lambda t: t.track_id)
    return out, joins
