"""Human-review queue: review sheets for flagged records and recorded decisions.

A flagged record (status ``review``) is excluded from the statistics until a
person accepts it. Decisions are stored in ``review_decisions.json`` next to the
records; the report applies them. Nothing here is automatic enforcement: the
tool produces audit statistics, and a reviewer decides about doubtful records.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import apply_homography
from .privacy import anonymized_crop
from .speed import fit_slope


def _zone_frames(est, n=4):
    idx = est.zone_idx
    if idx is None or len(idx) == 0:
        return []
    pick = np.unique(np.linspace(0, len(idx) - 1, n).round().astype(int))
    return [int(idx[i]) for i in pick]


def make_review_sheets(out, site, get_frames, out_dir) -> None:
    """Write one PNG per review item: anonymised crops + position-vs-time fit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {r.record_id: e for r, e in zip(out.records, out.estimates, strict=True)}
    tracks = {t.track_id: t for t in out.tracks}
    need = {}
    for item in out.review:
        est = by_id[item["record_id"]]
        tr = tracks[est.track_id]
        frame_obs = {o.frame: o for o in tr.observations}
        sel = [frame_obs[int(est.frames[i])] for i in _zone_frames(est)]
        need[item["record_id"]] = sel
    frames = get_frames({o.frame for sel in need.values() for o in sel})
    H = site.calibration.homography()
    axis = site.axis()
    for item in out.review:
        rid = item["record_id"]
        est = by_id[rid]
        sel = need[rid]
        fig = plt.figure(figsize=(8, 4.6), dpi=100)
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])
        for j, o in enumerate(sel[:4]):
            ax = fig.add_subplot(gs[0, j])
            f = frames.get(o.frame)
            if f is not None:
                crop = anonymized_crop(f, o.bbox)
                ax.imshow(crop[..., ::-1])
            ax.set_title(f"t={o.t:.2f}s", fontsize=8)
            ax.axis("off")
        ax = fig.add_subplot(gs[1, :])
        idx = est.zone_idx
        s = axis.to_sl(apply_homography(H, est.uv))[:, 0]
        ax.plot(est.t, s, ".", color="0.6", ms=3, label="all observations")
        ax.plot(est.t[idx], s[idx], ".", color="C0", ms=4, label="in zone")
        if len(idx) >= 2:
            b, a, _ = fit_slope(est.t[idx], s[idx], site.speed_config().fit)
            tt = est.t[idx]
            ax.plot(
                tt,
                a + b * (tt - tt.mean()),
                "-",
                color="C3",
                lw=1,
                label=f"fit {abs(b) * 3.6:.1f} km/h",
            )
        ax.axhspan(site.zone["start_m"], site.zone["end_m"], color="C2", alpha=0.08)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("along-road position (m)")
        ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"record {rid}: " + ", ".join(item["flags"]), fontsize=9)
        fig.tight_layout()
        name = f"record_{rid:06d}.png"
        fig.savefig(out_dir / name, metadata={"Software": None})
        plt.close(fig)
        item["sheet"] = f"review/{name}"


def load_decisions(run_dir) -> dict[int, str]:
    p = Path(run_dir) / "review_decisions.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {int(k): v for k, v in d.get("decisions", {}).items()}


def save_decisions(run_dir, decisions: dict[int, str], reviewer: str = "") -> None:
    p = Path(run_dir) / "review_decisions.json"
    body = {"reviewer": reviewer, "decisions": {str(k): v for k, v in sorted(decisions.items())}}
    p.write_text(json.dumps(body, indent=2) + "\n")


def interactive_review(run_dir, reviewer: str = "") -> dict[int, str]:
    """Step through review sheets: ``a`` accept, ``r`` reject, ``s`` skip, ``q`` quit."""
    import cv2

    run_dir = Path(run_dir)
    queue = json.loads((run_dir / "review_queue.json").read_text())["items"]
    decisions = load_decisions(run_dir)
    for item in queue:
        rid = int(item["record_id"])
        if rid in decisions:
            continue
        sheet = item.get("sheet")
        img = cv2.imread(str(run_dir / sheet)) if sheet else None
        if img is None:
            img = np.full((200, 700, 3), 255, np.uint8)
            cv2.putText(
                img,
                f"record {rid}: no sheet (run with --review-sheets)",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )
        cv2.imshow("roadspeed review (a=accept r=reject s=skip q=quit)", img)
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (ord("a"), ord("r"), ord("s"), ord("q")):
                break
        if k == ord("q"):
            break
        if k == ord("a"):
            decisions[rid] = "accept"
        elif k == ord("r"):
            decisions[rid] = "reject"
    cv2.destroyAllWindows()
    save_decisions(run_dir, decisions, reviewer)
    return decisions
