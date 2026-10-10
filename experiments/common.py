"""Shared machinery for the synthetic validation experiments.

Every experiment is a list of jobs (one simulated scenario + pipeline run each)
followed by an analysis that pools the per-vehicle errors per condition and writes
``results/<name>.json``, ``results/<name>_vehicles.csv`` and figures. All seeds are
fixed, floats are rounded on output, and jobs are sorted before aggregation, so
re-running produces byte-identical files on the same software stack.

Everything measured here is measured on simulated video.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Colours (validated reference palette, light mode): direction identity is fixed
# across every figure.
C_TOWARDS = "#2a78d6"
C_AWAY = "#eb6834"
C_THIRD = "#1baf7a"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e4e3df"
C_SURFACE = "#fcfcfb"


@dataclass
class Job:
    exp: str
    cond: str  # condition label (unique within the experiment)
    level: dict  # parameters that define the condition (for tables)
    seed: int
    duration: float
    preset: str = "baseline"
    overrides: dict = field(default_factory=dict)  # dotted scenario overrides
    site: dict = field(default_factory=dict)  # site_from_scenario keyword arguments
    post: list = field(default_factory=list)  # [{"name": str, "site": {...}}] post-hoc variants
    hero: str | None = None  # write an annotated example frame here
    keep_records: bool = False  # return the run's records + calibration scale samples

    def cost(self) -> float:
        fps = self.overrides.get("camera.fps", 30.0)
        w = self.overrides.get("camera.width", 1280)
        h = self.overrides.get("camera.height", 720)
        return self.duration * fps * (0.4 + w * h / (1280 * 720)) + 2.0 * len(self.post)


def _scenario(job: Job):
    from roadspeed.synth.presets import PRESETS

    cfg = PRESETS[job.preset](seed=job.seed, duration_s=job.duration)
    return cfg.replace(**job.overrides) if job.overrides else cfg


def _gsd(cfg) -> dict:
    """Ground sampling distance (metres per pixel, along the road) at the zone centre."""
    from roadspeed.geometry import homography_jacobian
    from roadspeed.synth import build_camera

    cam = build_camera(cfg)
    uv, _ = cam.project(np.array([[0.0, cfg.zone.center_m, 0.0]]))
    J = homography_jacobian(cam.image_to_ground_homography(), uv)[0]
    return {
        "gsd_along_m_per_px": float(abs(J[1, 1])),
        "tilt_deg": cam.tilt_deg,
        "focal_px": cam.focal_px,
    }


def run_job(job: Job) -> dict:
    import cv2

    cv2.setNumThreads(1)
    from roadspeed.evaluate import reevaluate, run_scenario, site_from_scenario

    t0 = time.time()
    cfg = _scenario(job)
    base = {"exp": job.exp, "cond": job.cond, "level": job.level, "seed": job.seed}
    try:
        site = site_from_scenario(cfg, **job.site)
    except ValueError as e:
        return {**base, "error": str(e), "geometry": _gsd(cfg)}
    res = run_scenario(cfg, site)
    out = {
        **base,
        "geometry": _gsd(cfg),
        "duration_s": cfg.duration_s,
        "metrics": res.metrics,
        "vehicles": res.vehicles,
        "summary_counts": res.out.summary["counts"],
        "calibration_scale_sd": float(
            np.nanstd(
                [
                    s
                    for s in res.out.summary["uncertainty"]["calibration_scale_samples"]
                    if s is not None
                ]
            )
        ),
        "post": {},
    }
    for v in job.post:
        try:
            m, veh = reevaluate(res, site_from_scenario(cfg, **v["site"]))
            out["post"][v["name"]] = {"metrics": m, "vehicles": veh}
        except ValueError as e:
            out["post"][v["name"]] = {"error": str(e)}
    if job.keep_records:
        out["records"] = [r.as_dict() for r in res.out.records]
        out["run_summary"] = res.out.summary
    if job.hero:
        _hero_frame(res, site, job.hero)
    out["seconds"] = round(time.time() - t0, 1)  # informational, not written to results
    return out


def _hero_frame(res, site, path) -> None:
    """Annotated example frame: zone, lanes, contact-point trails, measured vs true speed."""
    import cv2

    from roadspeed.geometry import apply_homography

    r = res.renderer
    H = site.calibration.homography()
    z0, z1 = site.zone["start_m"], site.zone["end_m"]
    truth = {v["vid"]: v for v in res.vehicles}
    # valid records and the ground-truth vehicle each was matched to (by measured speed)
    by_track: dict[int, tuple] = {}
    for rec, est in zip(res.out.records, res.out.estimates, strict=True):
        if rec.status != "valid":
            continue
        tv = next((v["true_kmh"] for v in truth.values() if v["meas_kmh"] == rec.speed_kmh), None)
        by_track[est.track_id] = (est, rec, tv)
    tracks = {t.track_id: t for t in res.out.tracks}

    def in_zone(o):
        u, v = 0.5 * (o.bbox[0] + o.bbox[2]), o.bbox[3]
        y = apply_homography(H, np.array([[u, v]]))[0, 1]
        return z0 + 3 <= y <= z1 - 3

    best, best_score = None, -1.0
    frames = sorted({o.frame for tid in by_track for o in tracks[tid].observations})
    for k in frames:
        cur = [(tid, o) for tid in by_track for o in tracks[tid].observations if o.frame == k]
        n_in = sum(in_zone(o) for _, o in cur)
        dirs = {by_track[tid][1].direction for tid, o in cur if in_zone(o)}
        score = n_in + 0.5 * len(dirs)
        if score > best_score:
            best, best_score = k, score
    if best is None:
        return
    frame, _ = r.render(best, annotate=False)
    t = r.frame_time(best)
    vis = frame.copy()
    # translucent zone on the road surface (not over vehicles)
    G = np.linalg.inv(H)
    xs = [ln["center_m"] - ln["width_m"] / 2 for ln in site.lanes] + [
        ln["center_m"] + ln["width_m"] / 2 for ln in site.lanes
    ]
    poly = apply_homography(
        G, np.array([[min(xs), z0], [max(xs), z0], [max(xs), z1], [min(xs), z1]])
    )
    zone = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(zone, np.round(poly).astype(np.int32), 1)
    road_only = (zone > 0) & ~r.vehicle_mask(t)
    tint = np.array([90, 200, 120], float)
    vis[road_only] = (0.72 * vis[road_only] + 0.28 * tint).astype(np.uint8)
    for k in range(4):
        a, b = poly[k], poly[(k + 1) % 4]
        if k in (0, 2):
            p1, p2 = tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int))
            cv2.line(vis, p1, p2, (70, 190, 110), 2, cv2.LINE_AA)
    for tid, (_est, rec, tv) in by_track.items():
        obs = [o for o in tracks[tid].observations if o.frame <= best]
        if not obs or obs[-1].frame != best:
            continue
        trail = [o for o in obs if not o.touches_border][-45:]
        for o in trail:
            c = (int(round(0.5 * (o.bbox[0] + o.bbox[2]))), int(round(o.bbox[3])))
            cv2.circle(vis, c, 2, (40, 220, 255), -1, cv2.LINE_AA)
        o = obs[-1]
        x1, y1, x2, y2 = (int(round(v)) for v in o.bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        label = f"{rec.speed_kmh:.1f} km/h" + (f"  true {tv:.1f}" if tv is not None else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(th + 8, y1)
        cv2.rectangle(vis, (x1, ty - th - 8), (x1 + tw + 8, ty), (24, 24, 24), -1)
        cv2.putText(
            vis,
            label,
            (x1 + 4, ty - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        "SIMULATED SCENE",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        "SIMULATED SCENE",
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def run_jobs(jobs: list[Job], workers: int) -> list[dict]:
    """Run jobs in parallel (most expensive first); return results sorted by job order."""
    import multiprocessing as mp

    order = sorted(range(len(jobs)), key=lambda i: -jobs[i].cost())
    results: list[dict | None] = [None] * len(jobs)
    t0 = time.time()
    if workers <= 1:
        for n, i in enumerate(order):
            results[i] = run_job(jobs[i])
            _progress(n + 1, len(jobs), t0, jobs[i])
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, maxtasksperchild=4) as pool:
            it = pool.imap_unordered(_run_indexed, [(i, jobs[i]) for i in order], chunksize=1)
            for n, (i, res) in enumerate(it):
                results[i] = res
                _progress(n + 1, len(jobs), t0, jobs[i])
    return results  # type: ignore[return-value]


def _run_indexed(arg):
    i, job = arg
    return i, run_job(job)


def _progress(n, total, t0, job):
    el = time.time() - t0
    print(f"  [{n}/{total}] {job.exp}:{job.cond} seed={job.seed}  ({el / 60:.1f} min)", flush=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def error_stats(e, seed: int = 0, n_boot: int = 1000) -> dict:
    """Bias, MAE, RMSE, P95(|e|) and bootstrap 95 % intervals (resampling vehicles)."""
    e = np.asarray([x for x in e if x is not None], float)
    if len(e) == 0:
        return {"n": 0}
    a = np.abs(e)
    out = {
        "n": int(len(e)),
        "bias": float(e.mean()),
        "mae": float(a.mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "p95_abs": float(np.percentile(a, 95)),
        "max_abs": float(a.max()),
    }
    if len(e) >= 10:
        rng = np.random.default_rng(np.random.SeedSequence([seed, len(e)]))
        idx = rng.integers(0, len(e), (n_boot, len(e)))
        eb = e[idx]
        ab = np.abs(eb)
        for k, vals in (
            ("bias", eb.mean(axis=1)),
            ("rmse", np.sqrt((eb**2).mean(axis=1))),
            ("p95_abs", np.percentile(ab, 95, axis=1)),
        ):
            out[k + "_ci"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


def pool_condition(results: list[dict], key: str = "main") -> dict:
    """Pool the runs (seeds) of one condition into summary metrics."""
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {
            "runs": 0,
            "failed_runs": len(results),
            "error": results[0].get("error") if results else None,
        }

    def part(r):
        return r if key == "main" else r["post"].get(key, {})

    parts = [part(r) for r in ok if "metrics" in part(r)]
    vehicles = [v for p in parts for v in p["vehicles"]]
    errs = [
        v["error_kmh"] for v in vehicles if v["status"] == "valid" and v["error_kmh"] is not None
    ]
    cover = [
        abs(v["error_kmh"]) <= 1.96 * v["u_kmh"]
        for v in vehicles
        if v["status"] == "valid" and v["error_kmh"] is not None and v["u_kmh"]
    ]
    outcomes = {
        k: sum(p["metrics"]["speed"]["outcomes"][k] for p in parts)
        for k in ("valid", "review_only", "rejected_only", "missed")
    }
    n_gt = sum(outcomes.values())
    false_valid = sum(p["metrics"]["speed"]["false_records"]["valid"] for p in parts)
    dup = sum(p["metrics"]["speed"]["duplicate_valid_records"] for p in parts)
    tr = [r["metrics"]["tracking"] for r in ok]
    tsum = {k: sum(t[k] for t in tr) for k in tr[0] if k != "mota"}
    tsum["mota"] = 1 - (
        tsum["false_negatives"] + tsum["false_positives"] + tsum["id_switches"]
    ) / max(tsum["gt_boxes"], 1)
    u_valid = [v["u_kmh"] for v in vehicles if v["status"] == "valid" and v["u_kmh"]]
    return {
        "runs": len(ok),
        "failed_runs": len(results) - len(ok),
        "gt_vehicles": n_gt,
        "outcomes": outcomes,
        "valid_rate": outcomes["valid"] / max(n_gt, 1),
        "false_valid_records": false_valid,
        "duplicate_valid_records": dup,
        "error_valid_kmh": error_stats(errs),
        "ci95_coverage": float(np.mean(cover)) if cover else None,
        "median_u_kmh": float(np.median(u_valid)) if u_valid else None,
        "tracking": tsum,
        "geometry": ok[0]["geometry"],
    }


def group_by_cond(results: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in results:
        out.setdefault(r["cond"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["seed"])
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _round(o, nd=4):
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_round(v, nd) for v in o]
    if isinstance(o, float | np.floating):
        f = float(o)
        return None if not np.isfinite(f) else round(f, nd)
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_round(obj), indent=2, sort_keys=True) + "\n")


def write_vehicles_csv(results: list[dict], path: Path, key: str = "main") -> None:
    rows = []
    for r in results:
        if "error" in r:
            continue
        p = r if key == "main" else r["post"].get(key, {})
        for v in p.get("vehicles", []):
            rows.append({"cond": r["cond"], "seed": r["seed"], **v})
    if not rows:
        return
    cols = [
        "cond",
        "seed",
        "vid",
        "class",
        "lane",
        "direction",
        "true_kmh",
        "meas_kmh",
        "error_kmh",
        "u_kmh",
        "status",
        "outcome",
        "flags",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow(
            {
                k: (
                    round(row[k], 4)
                    if isinstance(row[k], float)
                    else ("" if row[k] is None else row[k])
                )
                for k in cols
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue())


def job_meta(jobs: list[Job]) -> list[dict]:
    """Job definitions for the results file (what was run, with which seeds)."""
    keep = ("cond", "level", "seed", "duration", "preset", "overrides", "site")
    return [{k: v for k, v in asdict(j).items() if k in keep} for j in jobs]


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": C_SURFACE,
            "axes.facecolor": C_SURFACE,
            "savefig.facecolor": C_SURFACE,
            "axes.edgecolor": C_INK2,
            "axes.labelcolor": C_INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.color": C_INK2,
            "ytick.color": C_INK2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.color": C_GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "font.family": "DejaVu Sans",
            "lines.linewidth": 2.0,
            "svg.hashsalt": "roadspeed",
        }
    )
    return plt


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", metadata={"Software": None})


def workers_default() -> int:
    env = os.environ.get("ROADSPEED_WORKERS")
    if env:
        return int(env)
    return max(1, min(10, (os.cpu_count() or 2) - 2))


# ---------------------------------------------------------------------------
# Generic one-parameter sweep analysis
# ---------------------------------------------------------------------------


def analyze_sweep(
    name: str,
    title: str,
    jobs: list[Job],
    results: list[dict],
    x_key: str,
    x_label: str,
    rdir: Path,
    fdir: Path,
    log_x: bool = False,
    note: str = "",
) -> dict:
    """Pool per condition, write ``results/<name>.json`` + vehicles CSV + a figure."""
    plt = setup_matplotlib()
    groups = group_by_cond(results)
    conds = []
    for cond in dict.fromkeys(j.cond for j in jobs):
        rs = groups.get(cond, [])
        s = pool_condition(rs)
        s["cond"] = cond
        s["level"] = rs[0]["level"] if rs else {}
        conds.append(s)
    out = {
        "experiment": name,
        "title": title,
        "note": note,
        "conditions": conds,
        "jobs": job_meta(jobs),
    }
    write_json(out, rdir / f"{name}.json")
    write_vehicles_csv(results, rdir / f"{name}_vehicles.csv")
    # figure: error metrics and measurement rate vs the swept parameter
    xs, bias, rmse, p95, lo95, hi95, vr = [], [], [], [], [], [], []
    for c in conds:
        e = c.get("error_valid_kmh", {})
        if not e or e.get("n", 0) == 0:
            continue
        xs.append(c["level"][x_key])
        bias.append(e["bias"])
        rmse.append(e["rmse"])
        p95.append(e["p95_abs"])
        ci = e.get("p95_abs_ci", [e["p95_abs"], e["p95_abs"]])
        lo95.append(e["p95_abs"] - ci[0])
        hi95.append(ci[1] - e["p95_abs"])
        vr.append(c["valid_rate"])
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})
    if xs:
        ax[0].errorbar(
            xs,
            p95,
            yerr=[lo95, hi95],
            color=C_TOWARDS,
            marker="o",
            ms=5,
            capsize=3,
            label="P95 |error|",
        )
        ax[0].plot(xs, rmse, color=C_AWAY, marker="s", ms=5, label="RMSE")
        ax[0].plot(xs, bias, color=C_THIRD, marker="^", ms=5, label="bias (mean error)")
        ax[0].axhline(0, color=C_INK2, lw=0.8)
        ax[1].plot(xs, np.array(vr) * 100, color=C_TOWARDS, marker="o", ms=5)
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel("speed error (km/h)")
    ax[0].legend(loc="upper left")
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel("vehicles with a valid speed (%)")
    ax[1].set_ylim(0, 105)
    if log_x:
        for a in ax:
            a.set_xscale("log")
    fig.suptitle(f"{title} (simulated)", fontsize=10, color=C_INK)
    fig.tight_layout()
    save_fig(fig, fdir / f"{name}.png")
    plt.close(fig)
    return out
