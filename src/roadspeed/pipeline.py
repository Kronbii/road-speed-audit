"""End-to-end processing: frames -> detections -> tracks -> speeds -> records."""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .calibration import SiteConfig
from .detect import make_detector
from .records import VehicleRecord, dump_json, write_records_csv
from .speed import SpeedEstimate, estimate_speeds
from .tracking import Track, Tracker, TrackerConfig
from .uncertainty import UncertaintyConfig, calibration_draws, effective_click_sigma, propagate

FLAG_TEXT = {
    "few_points": "too few observations inside the measurement zone",
    "partial_zone": "track does not span enough of the zone (entered late or left early)",
    "implausible_speed": "speed outside the plausible range",
    "no_zone_points": "never observed inside the zone",
    "high_residual": "position-vs-time fit has large residuals (jumps in the contact point)",
    "gap": "long gap between observations (occlusion or missed detections)",
    "wide_blob": "box is wider on the ground than one vehicle (possible merge of two vehicles)",
    "lane_ambiguous": "lateral position does not fit one lane (lane change or merge)",
    "direction_mismatch": "travel direction does not match the lane's direction",
    "high_uncertainty": "speed uncertainty above the configured limit",
    "border": "vehicle touched the image border inside the zone",
    "mostly_merged": "most zone observations came from a blob shared with another vehicle",
    "split": "track was split at an identity jump; this record is one piece",
}


@dataclass
class RunOutput:
    records: list[VehicleRecord]
    estimates: list[SpeedEstimate]
    tracks: list[Track]
    summary: dict
    review: list[dict] = field(default_factory=list)


def _round_list(xs, nd=6):
    return [None if not np.isfinite(x) else round(float(x), nd) for x in xs]


def process_frames(
    frames: Iterable[tuple[int, float, np.ndarray]],
    site: SiteConfig,
    fps: float,
    prime: list[np.ndarray] | None = None,
    progress: Callable[[int], None] | None = None,
    source_name: str = "",
) -> RunOutput:
    """Run detection, tracking, speed estimation and uncertainty on a frame stream."""
    W, Hh = (int(v) for v in site.calibration.image_size)
    roi = site.roi_mask()
    detector = make_detector(site.detector, (W, Hh), roi)
    if prime:
        detector.prime(prime)
    tracker = Tracker(TrackerConfig(**site.tracker), fps=fps)
    n_frames = 0
    t_first, t_last = None, None
    for k, t, frame in frames:
        if frame.shape[1] != W or frame.shape[0] != Hh:
            raise ValueError(
                f"frame size {frame.shape[1]}x{frame.shape[0]} does not match the calibration "
                f"image size {W}x{Hh}"
            )
        dets = detector.detect(frame)
        tracker.update(dets, k, t)
        n_frames += 1
        t_first = t if t_first is None else t_first
        t_last = t
        if progress is not None:
            progress(k)
    tracks = tracker.finish()
    return finalize(tracks, site, fps, n_frames, t_first, t_last, source_name)


def finalize(tracks, site: SiteConfig, fps, n_frames, t_first, t_last, source_name="") -> RunOutput:
    cal = site.calibration
    H = cal.homography()
    axis = site.axis()
    lanes = site.lane_objs()
    scfg = site.speed_config()
    ucfg = UncertaintyConfig(**site.uncertainty)
    W, Hh = (int(v) for v in cal.image_size)
    cam = cal.camera() if scfg.contact_height_m else None
    cc = None if cam is None else cam.C
    estimates = [e for tr in tracks for e in estimate_speeds(tr, H, axis, scfg, lanes, (W, Hh), cc)]
    # drop tracks that never came near the zone (keeps the record list about the zone)
    estimates = [e for e in estimates if e.n_points > 0]
    # records are numbered in time order of their zone crossing
    estimates.sort(
        key=lambda e: (e.t_mid is None, e.t_mid if e.t_mid is not None else 0.0, e.track_id)
    )
    H_draws, C_draws, sig_px = calibration_draws(
        cal.img, cal.gnd, ucfg, (W, Hh), H, need_camera=bool(scfg.contact_height_m)
    )
    unc, scale = propagate(
        estimates, H, axis, scfg.fit, ucfg, H_draws, C_draws, cc, scfg.contact_height_m
    )
    t0 = None
    if site.start_time:
        t0 = _dt.datetime.fromisoformat(site.start_time)
    records = []
    review = []
    for rid, (est, u) in enumerate(zip(estimates, unc, strict=True), start=1):
        if u is not None and u["u_kmh"] > scfg.max_uncertainty_kmh:
            est.flags.append("high_uncertainty")
        status = est.status
        wall = None
        if t0 is not None and est.t_mid is not None:
            wall = (t0 + _dt.timedelta(seconds=est.t_mid)).isoformat(timespec="milliseconds")
        v = est.speed_kmh  # kept for rejected records too (diagnostics); status says why
        uk = None if u is None else u["u_kmh"]
        rec = VehicleRecord(
            record_id=rid,
            track_id=est.track_id,
            segment=est.segment,
            t_zone_in_s=est.t_zone_in,
            t_zone_out_s=est.t_zone_out,
            t_mid_s=est.t_mid,
            wall_time=wall,
            lane=est.lane,
            direction=est.direction,
            speed_kmh=v,
            u_kmh=uk,
            u_cal_kmh=None if u is None else u["u_cal_kmh"],
            u_rand_kmh=None if u is None else u["u_rand_kmh"],
            ci95_low_kmh=None if (u is None or v is None) else v - 1.96 * uk,
            ci95_high_kmh=None if (u is None or v is None) else v + 1.96 * uk,
            n_points=est.n_points,
            coverage=est.coverage,
            residual_m=est.residual_m,
            width_m=est.width_m,
            lateral_m=est.lateral_m,
            pixel_sigma_px=None if u is None else u["pixel_sigma_px"],
            label=est.label,
            status=status,
            flags=list(est.flags),
        )
        records.append(rec)
        if status == "review":
            review.append(
                {
                    "record_id": rec.record_id,
                    "t_mid_s": None if rec.t_mid_s is None else round(rec.t_mid_s, 3),
                    "lane": rec.lane,
                    "direction": rec.direction,
                    "speed_kmh": None if v is None else round(v, 2),
                    "flags": list(est.flags),
                    "reasons": [FLAG_TEXT.get(f, f) for f in est.flags],
                    "sheet": None,
                    "decision": None,
                }
            )
    rep = cal.report()
    counts = {s: sum(r.status == s for r in records) for s in ("valid", "review", "rejected")}
    duration = None if t_first is None else float(t_last - t_first + 1.0 / fps)
    summary = {
        "tool": "roadspeed",
        "version": __version__,
        "site": site.name,
        "source": Path(source_name).name if source_name else "",
        "fps": round(float(fps), 6),
        "frames": int(n_frames),
        "duration_s": None if duration is None else round(duration, 4),
        "start_time": site.start_time,
        "zone": site.zone,
        "lanes": site.lanes,
        "speed_limit_kmh": site.speed_limit_kmh,
        "n_tracks": len(tracks),
        "n_records": len(records),
        "counts": counts,
        "calibration": {
            "rms_px": rep["rms_px"],
            "max_px": rep["max_px"],
            "rms_m": rep["rms_m"],
            "n_points": rep["n_points"],
            "click_sigma_px_used": round(float(sig_px), 4),
            "ground_sigma_m": ucfg.ground_sigma_m,
        },
        "uncertainty": {
            "n_samples": ucfg.n_samples,
            "seed": ucfg.seed,
            "calibration_scale_samples": _round_list(scale, 6),
        },
        "click_sigma_px_from_residuals": round(
            float(effective_click_sigma(ucfg, cal.img, cal.gnd, H)), 4
        ),
    }
    return RunOutput(
        records=records, estimates=estimates, tracks=tracks, summary=summary, review=review
    )


def write_outputs(out: RunOutput, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(out.records, out_dir / "records.csv")
    dump_json(
        {"summary": out.summary, "records": [r.as_dict() for r in out.records]},
        out_dir / "records.json",
    )
    dump_json(out.summary, out_dir / "summary.json")
    dump_json({"items": out.review}, out_dir / "review_queue.json")


# ---------------------------------------------------------------------------
# Video files
# ---------------------------------------------------------------------------


def video_info(path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def iter_video(path, fps: float, timestamps: str = "fps", max_frames: int | None = None):
    """Yield ``(index, time_s, frame)``. ``timestamps="container"`` uses decoder times."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    k = 0
    try:
        while max_frames is None or k < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if timestamps == "container":
                t = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            else:
                t = k / fps
            yield k, t, frame
            k += 1
    finally:
        cap.release()


def sample_frames(path, n: int, max_frames: int | None = None) -> list[np.ndarray]:
    """``n`` frames spread evenly over the video (for the median background)."""
    import cv2

    info = video_info(path)
    total = info["frames"] if max_frames is None else min(info["frames"], max_frames)
    if total <= 0 or n <= 0:
        return []
    idx = np.unique(np.linspace(0, total - 1, n).astype(int))
    cap = cv2.VideoCapture(str(path))
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def read_frames_by_index(path, indices: set[int]) -> dict[int, np.ndarray]:
    """Sequentially read the frames with the given indices (for review sheets)."""
    import cv2

    want = sorted(indices)
    out = {}
    if not want:
        return out
    cap = cv2.VideoCapture(str(path))
    k = 0
    j = 0
    while j < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if k == want[j]:
            out[k] = f
            j += 1
        k += 1
    cap.release()
    return out


def run_video(
    path,
    site: SiteConfig,
    out_dir=None,
    review_sheets: bool = False,
    max_frames: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> RunOutput:
    info = video_info(path)
    fps = float(site.fps or info["fps"])
    if fps <= 0:
        raise ValueError("unknown frame rate: set 'fps' in the site config")
    prime = (
        sample_frames(path, site.prime_frames, max_frames)
        if site.detector.get("backend", "bgsub") == "bgsub"
        else None
    )
    out = process_frames(
        iter_video(path, fps, site.timestamps, max_frames), site, fps, prime, progress, str(path)
    )
    if out_dir is not None:
        if review_sheets and out.review:
            from .review import make_review_sheets

            make_review_sheets(
                out, site, lambda idx: read_frames_by_index(path, idx), Path(out_dir) / "review"
            )
        write_outputs(out, out_dir)
    return out
