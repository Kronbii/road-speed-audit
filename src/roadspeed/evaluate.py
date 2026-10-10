"""Evaluation against synthetic ground truth (simulation only).

* :func:`site_from_scenario` builds the site configuration a field team would
  produce for a simulated scene: calibration points on lane-marking corners
  inside the zone, clicked with a chosen pixel error and "surveyed" with a chosen
  tape error.
* :func:`run_scenario` renders the scene, runs the full pipeline on the frames
  (exactly as on a video file) and scores the result.
* :func:`evaluate` matches tracks to ground-truth vehicles frame by frame (IoU
  with Hungarian assignment) for tracking metrics, and records to vehicles for
  speed errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .calibration import Calibration, SiteConfig
from .pipeline import RunOutput, finalize, process_frames
from .synth import Renderer, ScenarioConfig, build_camera
from .tracking import iou_matrix


def calibration_ground_points(cfg: ScenarioConfig, layout: str = "zone") -> np.ndarray:
    """Surveyable points (lane-line positions) for a simulated site.

    ``zone``: road edges and centre line at the zone start, middle and end (9 points).
    ``compact``: 4 points close together at the near end of the zone (a poor layout).
    ``wide``: like ``zone`` but extending 10 m beyond both ends of the zone.
    """
    road, z = cfg.road, cfg.zone
    hw = road.width_m / 2
    xs = [-hw, 0.0, hw] if road.n_lanes > 1 else [-hw, hw]
    if layout == "zone":
        ys = [z.start_m, z.center_m, z.end_m]
    elif layout == "wide":
        ys = [z.start_m - 10, z.center_m, z.end_m + 10]
    elif layout == "compact":
        ys = [z.start_m, z.start_m + 6.0]
        xs = [-hw, hw]
    elif layout == "near_half":
        ys = [z.start_m, z.start_m + 0.25 * (z.end_m - z.start_m), z.center_m]
    else:
        raise ValueError(f"unknown calibration layout {layout!r}")
    return np.array([[x, y] for y in ys for x in xs], dtype=float)


def scenario_lanes(cfg: ScenarioConfig) -> list[dict]:
    road = cfg.road
    out = []
    c = road.lane_centers()
    dirs = road.lane_directions()
    na = nt = 0
    for i in range(road.n_lanes):
        if dirs[i] > 0:
            na += 1
            name = f"away-{na}"
        else:
            nt += 1
            name = f"towards-{nt}"
        out.append(
            {
                "name": name,
                "center_m": float(c[i]),
                "width_m": road.lane_width_m,
                "direction": "away" if dirs[i] > 0 else "towards",
            }
        )
    return out


def site_from_scenario(
    cfg: ScenarioConfig,
    click_sigma_px: float = 0.7,
    ground_sigma_m: float = 0.02,
    layout: str = "zone",
    seed: int = 0,
    assumed_click_sigma_px: float | None = None,
    assumed_ground_sigma_m: float | None = None,
    speed: dict | None = None,
    detector: dict | None = None,
    tracker: dict | None = None,
    uncertainty: dict | None = None,
    speed_limit_kmh: float | None = 50.0,
    round_clicks: bool = True,
) -> SiteConfig:
    """The site config a careful field calibration would give for this simulated scene.

    Clicks are perturbed by ``click_sigma_px`` and rounded to whole pixels (as in a
    GUI), surveyed coordinates by ``ground_sigma_m``. The uncertainty model is told
    ``assumed_*`` sigmas (default: the true ones).
    """
    cam = build_camera(cfg)
    gnd_true = calibration_ground_points(cfg, layout)
    uv, depth = cam.project(np.column_stack([gnd_true, np.zeros(len(gnd_true))]))
    if (
        np.any(depth <= 0)
        or np.any(uv[:, 0] < 0)
        or np.any(uv[:, 0] > cfg.camera.width - 1)
        or np.any(uv[:, 1] < 0)
        or np.any(uv[:, 1] > cfg.camera.height - 1)
    ):
        raise ValueError("calibration points fall outside the image for this camera placement")
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 5, seed]))
    clicks = uv + rng.normal(0, click_sigma_px, uv.shape) if click_sigma_px > 0 else uv.copy()
    if round_clicks:
        clicks = np.round(clicks)
    gnd = gnd_true + (rng.normal(0, ground_sigma_m, gnd_true.shape) if ground_sigma_m > 0 else 0.0)
    unc = {
        "click_sigma_px": assumed_click_sigma_px
        if assumed_click_sigma_px is not None
        else max(click_sigma_px, 0.3),
        "ground_sigma_m": assumed_ground_sigma_m
        if assumed_ground_sigma_m is not None
        else max(ground_sigma_m, 0.005),
    }
    unc.update(uncertainty or {})
    cal = Calibration(
        image_points=np.round(clicks, 3).tolist(),
        ground_points=np.round(gnd, 4).tolist(),
        image_size=[cfg.camera.width, cfg.camera.height],
        click_sigma_px=unc["click_sigma_px"],
        ground_sigma_m=unc["ground_sigma_m"],
        note="simulated calibration",
    )
    return SiteConfig(
        name=f"synthetic-seed{cfg.seed}",
        calibration=cal,
        zone={"start_m": cfg.zone.start_m, "end_m": cfg.zone.end_m},
        lanes=scenario_lanes(cfg),
        speed_limit_kmh=speed_limit_kmh,
        detector=detector or {"backend": "bgsub"},
        tracker=tracker or {},
        speed=speed or {},
        uncertainty=unc,
        prime_frames=25,
    )


@dataclass
class ScenarioResult:
    metrics: dict
    vehicles: list[dict]
    out: RunOutput
    renderer: Renderer | None = None
    anns: dict | None = None
    cfg: ScenarioConfig | None = None
    frame_match: tuple | None = None


def run_scenario(
    cfg: ScenarioConfig,
    site: SiteConfig | None = None,
    prime_n: int = 25,
    progress=None,
) -> ScenarioResult:
    """Render a scenario, run the pipeline on its frames and evaluate."""
    r = Renderer(cfg)
    site = site or site_from_scenario(cfg)
    prime = None
    if site.detector.get("backend", "bgsub") == "bgsub" and prime_n > 0:
        idx = np.unique(np.linspace(0, r.n_frames - 1, prime_n).astype(int))
        prime = [r.render(int(k), annotate=False)[0] for k in idx]
    anns: dict[int, list] = {}

    def frames():
        for k in range(r.n_frames):
            f, a = r.render(k, annotate=True)
            anns[k] = a
            yield k, r.frame_time(k), f

    out = process_frames(frames(), site, r.fps, prime, progress, source_name="synthetic")
    metrics, vehicles = evaluate(out, r, anns, cfg)
    return ScenarioResult(
        metrics=metrics, vehicles=vehicles, out=out, renderer=r, anns=anns, cfg=cfg
    )


def reevaluate(res: ScenarioResult, site: SiteConfig):
    """Re-run speed estimation and scoring on the same tracks with another site config.

    Used for post-hoc sweeps that do not change detection or tracking (calibration
    noise, calibration layout, contact-point model).
    """
    s = res.out.summary
    fps = s["fps"]
    out2 = finalize(
        res.out.tracks, site, fps, s["frames"], 0.0, s["duration_s"] - 1.0 / fps, "synthetic"
    )
    return evaluate(out2, res.renderer, res.anns, res.cfg)


def _summ(e: np.ndarray) -> dict:
    if len(e) == 0:
        return {"n": 0, "bias": None, "mae": None, "rmse": None, "p95_abs": None, "max_abs": None}
    a = np.abs(e)
    return {
        "n": int(len(e)),
        "bias": float(np.mean(e)),
        "mae": float(np.mean(a)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "p95_abs": float(np.percentile(a, 95)),
        "max_abs": float(a.max()),
    }


def evaluate(
    out: RunOutput,
    renderer: Renderer,
    anns: dict,
    cfg: ScenarioConfig,
    min_iou: float = 0.3,
    frame_match=None,
):
    """Score a run. Returns ``(metrics, vehicles)``.

    ``frame_match`` (``(tracking, frame_track_gt)`` from an earlier call on the same
    tracks) skips the frame-level matching when only speed estimation changed.
    """
    gt = renderer.ground_truth_vehicles()
    gt_by_id = {g["vid"]: g for g in gt}
    if frame_match is None:
        frame_match = match_frames(out, renderer, anns, cfg, gt_by_id, min_iou)
    tracking, frame_track_gt = frame_match
    metrics, vehicles = score_records(out, gt, gt_by_id, frame_track_gt)
    metrics = {"tracking": tracking, **metrics}
    return metrics, vehicles


def match_frames(
    out: RunOutput, renderer: Renderer, anns: dict, cfg: ScenarioConfig, gt_by_id, min_iou=0.3
):
    """Frame-by-frame IoU matching of track observations to ground-truth boxes."""
    z = cfg.zone
    # tracking is scored where the pipeline is expected to track: the zone plus the
    # margins used for the region of interest; elsewhere boxes are "don't care"
    track_lo, track_hi = z.start_m - 25.0, z.end_m + 60.0
    # ---------------- frame-level matching (tracking quality)
    obs_by_frame: dict[int, list] = {}
    for tr in out.tracks:
        for o in tr.observations:
            obs_by_frame.setdefault(o.frame, []).append((tr.track_id, o.bbox))
    frame_track_gt: dict[tuple, int] = {}
    gt_seq: dict[int, list] = {}  # vid -> [(frame, track or None, in_zone)]
    n_gt = n_fn = n_fp = 0
    for k in sorted(anns):
        t = renderer.frame_time(k)
        a_req, a_dc = [], []
        for a in anns[k]:
            if a.bbox_visible is None:
                continue
            y = float(renderer.vehicles[a.vid].y_at(t))
            h = a.bbox_visible[3] - a.bbox_visible[1]
            if a.visibility >= 0.3 and h >= 10 and track_lo <= y <= track_hi:
                a_req.append(a)
            else:
                a_dc.append(a)
        obs = obs_by_frame.get(k, [])
        matched_t = set()
        m = {}
        if a_req and obs:
            iou = iou_matrix(
                np.array([a.bbox_visible for a in a_req], float), np.array([b for _, b in obs])
            )
            cost = np.where(iou >= min_iou, 1 - iou, 1e6)
            rr, cc = linear_sum_assignment(cost)
            for i, j in zip(rr, cc, strict=True):
                if cost[i, j] < 1e5:
                    m[i] = obs[j][0]
                    matched_t.add(j)
        for i, a in enumerate(a_req):
            v = renderer.vehicles[a.vid]
            y = float(v.y_at(t))
            in_zone = z.start_m <= y <= z.end_m
            tid = m.get(i)
            gt_seq.setdefault(a.vid, []).append((k, tid, in_zone))
            if tid is not None:
                frame_track_gt[(k, tid)] = a.vid
        n_gt += len(a_req)
        n_fn += len(a_req) - len(m)
        # unmatched observations overlapping a don't-care box are not false positives
        for j, (_, b) in enumerate(obs):
            if j in matched_t:
                continue
            if (
                a_dc
                and iou_matrix(np.array([b]), np.array([a.bbox_visible for a in a_dc], float)).max()
                >= min_iou
            ):
                continue
            n_fp += 1
    idsw = frag = zone_idsw = zone_frag = 0
    mostly_tracked = 0
    zone_fully_tracked = 0
    n_traj = 0
    n_zone_traj = 0
    for vid, seq in gt_seq.items():
        if not gt_by_id[vid]["in_window"]:
            continue
        n_traj += 1
        ids = [tid for _, tid, _ in seq]
        last = None
        tracked_before = False
        gap = False
        for tid in ids:
            if tid is None:
                if tracked_before:
                    gap = True
                continue
            if last is not None and tid != last:
                idsw += 1
            if gap:
                frag += 1
                gap = False
            last = tid
            tracked_before = True
        frac = np.mean([tid is not None for tid in ids]) if ids else 0.0
        mostly_tracked += frac >= 0.8
        zids = [tid for _, tid, inz in seq if inz]
        if zids:
            n_zone_traj += 1
            zl = None
            zb = False
            zg = False
            for tid in zids:
                if tid is None:
                    zg = zb or zg
                    continue
                if zl is not None and tid != zl:
                    zone_idsw += 1
                if zg:
                    zone_frag += 1
                    zg = False
                zl = tid
                zb = True
            zone_fully_tracked += all(t_ is not None for t_ in zids) and len(set(zids)) == 1
    tracking = {
        "gt_boxes": n_gt,
        "false_negatives": n_fn,
        "false_positives": n_fp,
        "id_switches": idsw,
        "fragmentations": frag,
        "mota": float(1 - (n_fn + n_fp + idsw) / max(n_gt, 1)),
        "trajectories": n_traj,
        "mostly_tracked": int(mostly_tracked),
        "zone_trajectories": n_zone_traj,
        "zone_id_switches": zone_idsw,
        "zone_fragmentations": zone_frag,
        "zone_fully_tracked_one_id": int(zone_fully_tracked),
    }
    return tracking, frame_track_gt


def score_records(out: RunOutput, gt, gt_by_id, frame_track_gt):
    """Match records to ground-truth vehicles (majority vote over zone frames) and score."""
    recs_by_gt: dict[int, list] = {}
    false_records = {"valid": 0, "review": 0, "rejected": 0}
    for rec, est in zip(out.records, out.estimates, strict=True):
        # majority vote over the frames this record's speed was measured on
        frames = est.frames[est.zone_idx] if est.zone_idx is not None and len(est.zone_idx) else []
        votes: dict[int, int] = {}
        for f in frames:
            vid = frame_track_gt.get((int(f), est.track_id))
            if vid is not None:
                votes[vid] = votes.get(vid, 0) + 1
        if not votes or max(votes.values()) < 0.5 * len(frames):
            false_records[rec.status] += 1
            continue
        vid = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        if not gt_by_id[vid]["in_window"]:
            continue
        recs_by_gt.setdefault(vid, []).append(rec)
    vehicles = []
    errs = {"valid": [], "review": []}
    covered = []
    dup = 0
    outcome_counts = {"valid": 0, "review_only": 0, "rejected_only": 0, "missed": 0}
    for g in gt:
        if not g["in_window"]:
            continue
        recs = recs_by_gt.get(g["vid"], [])
        valid = [r for r in recs if r.status == "valid"]
        review = [r for r in recs if r.status == "review"]
        if valid:
            outcome = "valid"
            dup += len(valid) - 1
        elif review:
            outcome = "review_only"
        elif recs:
            outcome = "rejected_only"
        else:
            outcome = "missed"
        outcome_counts[outcome] += 1
        best = (valid or review or recs or [None])[0]
        row = {
            "vid": g["vid"],
            "class": g["class"],
            "lane": g["lane"],
            "direction": g["direction"],
            "true_kmh": g["speed_kmh"],
            "outcome": outcome,
            "meas_kmh": None if best is None else best.speed_kmh,
            "u_kmh": None if best is None else best.u_kmh,
            "status": None if best is None else best.status,
            "flags": "" if best is None else ";".join(best.flags),
            "error_kmh": None,
        }
        if best is not None and best.speed_kmh is not None:
            row["error_kmh"] = best.speed_kmh - g["speed_kmh"]
            if best.status in errs:
                errs[best.status].append(row["error_kmh"])
            if best.status == "valid" and best.u_kmh:
                covered.append(abs(row["error_kmh"]) <= 1.96 * best.u_kmh)
        vehicles.append(row)
    n_in = sum(outcome_counts.values())
    speed = {
        "gt_vehicles": n_in,
        "outcomes": outcome_counts,
        "valid_rate": outcome_counts["valid"] / max(n_in, 1),
        "duplicate_valid_records": dup,
        "false_records": false_records,
        "error_valid": _summ(np.array(errs["valid"])),
        "error_review": _summ(np.array(errs["review"])),
        "ci95_coverage_valid": float(np.mean(covered)) if covered else None,
        "median_u_valid": float(
            np.median([v["u_kmh"] for v in vehicles if v["status"] == "valid" and v["u_kmh"]])
        )
        if any(v["status"] == "valid" and v["u_kmh"] for v in vehicles)
        else None,
    }
    metrics = {"speed": speed, "calibration": out.summary["calibration"]}
    return metrics, vehicles
