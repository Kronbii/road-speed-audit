"""Multi-object tracker in the SORT / ByteTrack family.

* A constant-velocity Kalman filter on the box ``(cx, cy, w, h)``.
* Hungarian assignment on ``1 - IoU`` between predicted tracks and detections,
  in ByteTrack's two stages (confident detections first, then low-score ones for
  tracks that were matched in the previous frame), plus a centre-distance
  fallback for small, fast boxes whose predicted IoU is zero.
* Tracks without a match are kept "lost" and coasted on the motion model for up
  to ``max_lost_s`` seconds, so a short occlusion does not end the track.

Only frames with a real detection are stored as observations; coasted
predictions are never used for speed estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect.base import Detection


class TrackState(Enum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class TrackerConfig:
    high_score: float = 0.5
    low_score: float = 0.1
    new_track_score: float = 0.5
    iou_first: float = 0.2  # minimum IoU for a stage-1 match
    iou_second: float = 0.4  # minimum IoU for low-score (stage-2) matches
    iou_tentative: float = 0.3
    center_gate: float = 0.7  # centre distance / predicted box size for the fallback match
    center_gate_min_px: float = 24.0  # ... with the box size floored at this many pixels
    min_hits: int = 3
    max_lost_s: float = 1.0
    # merged-blob handling: a detection that contains >= 2 confirmed predictions is a
    # merge; it updates only the track whose predicted bottom edge it matches (the
    # vehicle in front), the others coast as occluded for up to max_occluded_s.
    handle_merges: bool = True
    merge_containment: float = 0.6
    merge_bottom_tol: float = 0.25  # |bottom difference| / predicted height
    max_occluded_s: float = 3.0
    std_pos: float = 1.0 / 20
    std_vel: float = 1.0 / 160
    std_meas: float = 1.0 / 20


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes ``a (N,4)`` and ``b (M,4)`` given as ``x1 y1 x2 y2``."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)])


def _xywh_to_xyxy(s):
    cx, cy, w, h = s[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class KalmanBox:
    """Constant-velocity Kalman filter on ``(cx, cy, w, h)`` with size-scaled noise."""

    def __init__(self, bbox, cfg: TrackerConfig):
        self.cfg = cfg
        z = _xyxy_to_xywh(bbox)
        self.x = np.concatenate([z, np.zeros(4)])
        s = np.array([z[2], z[3], z[2], z[3]])
        self.P = np.diag(np.concatenate([(2 * cfg.std_pos * s) ** 2, (10 * cfg.std_vel * s) ** 2]))

    def _scale(self):
        w, h = max(self.x[2], 1.0), max(self.x[3], 1.0)
        return np.array([w, h, w, h])

    def predict(self, dt: float = 1.0) -> None:
        F = np.eye(8)
        F[:4, 4:] = dt * np.eye(4)
        s = self._scale()
        q = np.concatenate([(self.cfg.std_pos * s) ** 2, (self.cfg.std_vel * s) ** 2]) * dt
        self.x = F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.P = F @ self.P @ F.T + np.diag(q)

    def update(self, bbox) -> None:
        z = _xyxy_to_xywh(bbox)
        Hm = np.zeros((4, 8))
        Hm[:4, :4] = np.eye(4)
        R = np.diag((self.cfg.std_meas * self._scale()) ** 2)
        S = Hm @ self.P @ Hm.T + R
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - Hm @ self.x)
        self.P = (np.eye(8) - K @ Hm) @ self.P

    @property
    def bbox(self) -> np.ndarray:
        return _xywh_to_xyxy(self.x)


@dataclass
class Observation:
    frame: int
    t: float
    bbox: np.ndarray
    score: float
    touches_border: bool
    label: str = "vehicle"
    merged: bool = False  # box synthesised from a merged blob (bottom edge only)
    bottom_width_px: float | None = None


@dataclass
class Track:
    track_id: int
    kf: KalmanBox
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    last_frame: int = 0
    occluded_until: int = -1  # frame index until which a lost track is kept alive
    observations: list[Observation] = field(default_factory=list)

    @property
    def bbox(self) -> np.ndarray:
        return self.kf.bbox

    @property
    def label(self) -> str:
        labels = [o.label for o in self.observations]
        return max(set(labels), key=labels.count) if labels else "vehicle"


class Tracker:
    def __init__(self, cfg: TrackerConfig | None = None, fps: float = 30.0):
        self.cfg = cfg or TrackerConfig()
        self.fps = fps
        self.max_lost = max(1, int(round(self.cfg.max_lost_s * fps)))
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 1
        self._frame = None

    # ------------------------------------------------------------------ utils
    def _match(self, tracks, dets, min_iou):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        ious = iou_matrix(np.array([t.bbox for t in tracks]), np.array([d.bbox for d in dets]))
        cost = 1.0 - ious
        cost[ious < min_iou] = 1e6
        r, c = linear_sum_assignment(cost)
        matches = [(i, j) for i, j in zip(r, c, strict=True) if cost[i, j] < 1e5]
        mt = {i for i, _ in matches}
        md = {j for _, j in matches}
        return (
            matches,
            [i for i in range(len(tracks)) if i not in mt],
            [j for j in range(len(dets)) if j not in md],
        )

    def _match_center(self, tracks, dets):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        tb = np.array([t.bbox for t in tracks])
        db = np.array([d.bbox for d in dets])
        tc = (tb[:, :2] + tb[:, 2:]) / 2
        dc = (db[:, :2] + db[:, 2:]) / 2
        ts = np.maximum(tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1])
        ds = np.maximum(db[:, 2] - db[:, 0], db[:, 3] - db[:, 1])
        dist = np.linalg.norm(tc[:, None, :] - dc[None, :, :], axis=2) / np.maximum(
            ts[:, None], self.cfg.center_gate_min_px
        )
        ratio = ds[None, :] / np.maximum(ts[:, None], 1.0)
        cost = dist.copy()
        cost[(dist > self.cfg.center_gate) | (ratio < 0.5) | (ratio > 2.0)] = 1e6
        r, c = linear_sum_assignment(cost)
        matches = [(i, j) for i, j in zip(r, c, strict=True) if cost[i, j] < 1e5]
        mt = {i for i, _ in matches}
        md = {j for _, j in matches}
        return (
            matches,
            [i for i in range(len(tracks)) if i not in mt],
            [j for j in range(len(dets)) if j not in md],
        )

    def _apply(
        self, track: Track, det: Detection, frame: int, t: float, merged: bool = False
    ) -> None:
        track.kf.update(det.bbox)
        track.hits += 1
        track.last_frame = frame
        track.observations.append(
            Observation(
                frame,
                t,
                det.bbox.copy(),
                det.score,
                det.touches_border,
                det.label,
                merged,
                None if merged else det.bottom_width_px,
            )
        )
        if track.state == TrackState.TENTATIVE and track.hits >= self.cfg.min_hits:
            track.state = TrackState.CONFIRMED
        elif track.state == TrackState.LOST:
            track.state = TrackState.CONFIRMED

    def _resolve_merges(self, active, high, frame, t):
        """Handle detections that cover two or more confirmed tracks (merged blobs)."""
        conf = [
            tr
            for tr in active
            if tr.state == TrackState.CONFIRMED or tr.hits >= 2 * self.cfg.min_hits
        ]
        if len(conf) < 2 or not high:
            return active, high
        tb = np.array([tr.bbox for tr in conf])
        db = np.array([d.bbox for d in high])
        x1 = np.maximum(tb[:, None, 0], db[None, :, 0])
        y1 = np.maximum(tb[:, None, 1], db[None, :, 1])
        x2 = np.minimum(tb[:, None, 2], db[None, :, 2])
        y2 = np.minimum(tb[:, None, 3], db[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_t = np.maximum((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1]), 1e-6)
        cont = inter / area_t[:, None]
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for j in range(len(high)):
            members = [
                i
                for i in np.nonzero(cont[:, j] >= self.cfg.merge_containment)[0]
                if i not in used_tracks
            ]
            if len(members) < 2:
                continue
            d = high[j]
            h = np.maximum(tb[members, 3] - tb[members, 1], 1.0)
            err = np.abs(tb[members, 3] - d.bbox[3]) / h
            k = int(np.argmin(err))
            front = members[k]
            if err[k] <= self.cfg.merge_bottom_tol:
                p = tb[front]
                box = np.array([p[0], d.bbox[3] - (p[3] - p[1]), p[2], d.bbox[3]])
                syn = Detection(
                    bbox=box,
                    score=d.score,
                    label=d.label,
                    area_px=d.area_px,
                    touches_border=d.touches_border,
                )
                self._apply(conf[front], syn, frame, t, merged=True)
                used_tracks.add(front)
            for i in members:
                if i != front or err[k] > self.cfg.merge_bottom_tol:
                    tr = conf[i]
                    tr.occluded_until = frame + int(round(self.cfg.max_occluded_s * self.fps))
                    if tr.state == TrackState.CONFIRMED:
                        tr.state = TrackState.LOST
                    used_tracks.add(i)
            used_dets.add(j)
        if not used_dets:
            return active, high
        used = {id(conf[i]) for i in used_tracks}
        return [tr for tr in active if id(tr) not in used], [
            d for j, d in enumerate(high) if j not in used_dets
        ]

    # ----------------------------------------------------------------- update
    def update(self, detections: list[Detection], frame: int, t: float) -> list[Track]:
        cfg = self.cfg
        dt = 1.0 if self._frame is None else float(frame - self._frame)
        self._frame = frame
        for tr in self.tracks:
            tr.kf.predict(dt)

        high = [d for d in detections if d.score >= cfg.high_score]
        low = [d for d in detections if cfg.low_score <= d.score < cfg.high_score]
        active = [t_ for t_ in self.tracks if t_.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t_ for t_ in self.tracks if t_.state == TrackState.TENTATIVE]
        if cfg.handle_merges:
            active, high = self._resolve_merges(active, high, frame, t)

        # stage 1: confident detections vs confirmed + lost tracks
        m1, ut1, ud1 = self._match(active, high, cfg.iou_first)
        for i, j in m1:
            self._apply(active[i], high[j], frame, t)
        rem_tracks = [active[i] for i in ut1]
        rem_high = [high[j] for j in ud1]

        # stage 1b: centre-distance fallback for small / fast boxes
        m1b, ut1b, ud1b = self._match_center(rem_tracks, rem_high)
        for i, j in m1b:
            self._apply(rem_tracks[i], rem_high[j], frame, t)
        rem_tracks = [rem_tracks[i] for i in ut1b]
        rem_high = [rem_high[j] for j in ud1b]

        # stage 2: low-score detections vs tracks matched in the previous frame
        recent = [t_ for t_ in rem_tracks if t_.state == TrackState.CONFIRMED]
        m2, ut2, _ = self._match(recent, low, cfg.iou_second)
        for i, j in m2:
            self._apply(recent[i], low[j], frame, t)
        matched2 = {id(recent[i]) for i, _ in m2}
        for t_ in rem_tracks:
            if id(t_) not in matched2 and t_.state == TrackState.CONFIRMED:
                t_.state = TrackState.LOST

        # stage 3: tentative tracks vs leftover confident detections
        m3, ut3, ud3 = self._match(tentative, rem_high, cfg.iou_tentative)
        for i, j in m3:
            self._apply(tentative[i], rem_high[j], frame, t)
        tent_left = [tentative[i] for i in ut3]
        rem_high = [rem_high[j] for j in ud3]
        m3b, ut3b, ud3b = self._match_center(tent_left, rem_high)
        for i, j in m3b:
            self._apply(tent_left[i], rem_high[j], frame, t)
        for i in ut3b:
            tent_left[i].state = TrackState.REMOVED
        rem_high = [rem_high[j] for j in ud3b]

        # new tracks
        for d in rem_high:
            if d.score < cfg.new_track_score:
                continue
            tr = Track(track_id=self._next_id, kf=KalmanBox(d.bbox, cfg), last_frame=frame)
            tr.observations.append(
                Observation(
                    frame,
                    t,
                    d.bbox.copy(),
                    d.score,
                    d.touches_border,
                    d.label,
                    False,
                    d.bottom_width_px,
                )
            )
            if cfg.min_hits <= 1:
                tr.state = TrackState.CONFIRMED
            self._next_id += 1
            self.tracks.append(tr)

        # retire lost tracks
        keep = []
        for tr in self.tracks:
            if (
                tr.state == TrackState.LOST
                and frame - tr.last_frame > self.max_lost
                and frame > tr.occluded_until
            ):
                tr.state = TrackState.REMOVED
            if tr.state == TrackState.REMOVED:
                if tr.hits >= cfg.min_hits:
                    self.finished.append(tr)
            else:
                keep.append(tr)
        self.tracks = keep
        return [
            t_ for t_ in self.tracks if t_.state == TrackState.CONFIRMED and t_.last_frame == frame
        ]

    def finish(self) -> list[Track]:
        """All tracks that were ever confirmed, ordered by id."""
        out = list(self.finished)
        out += [t for t in self.tracks if t.hits >= self.cfg.min_hits]
        self.tracks = []
        self.finished = []
        return sorted(out, key=lambda t: t.track_id)
