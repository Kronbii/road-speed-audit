"""E01 - baseline accuracy: two-lane road, 6 m pole, 720p, 30 fps, zone 20-50 m."""

from __future__ import annotations

import numpy as np

from common import (
    C_AWAY,
    C_INK2,
    C_TOWARDS,
    Job,
    error_stats,
    group_by_cond,
    job_meta,
    pool_condition,
    save_fig,
    setup_matplotlib,
    write_json,
    write_vehicles_csv,
)

NAME = "e01_baseline"


def jobs(quick: bool, fdir) -> list[Job]:
    seeds = [101, 102] if quick else list(range(101, 111))
    dur = 45.0 if quick else 120.0
    return [
        Job(NAME, "baseline", {}, s, dur, hero=str(fdir / "hero_frame.png") if s == 101 else None)
        for s in seeds
    ]


def analyze(js, results, rdir, fdir, **_):
    plt = setup_matplotlib()
    rs = group_by_cond(results)["baseline"]
    pooled = pool_condition(rs)
    veh = [v for r in rs for v in r["vehicles"]]
    valid = [v for v in veh if v["status"] == "valid" and v["error_kmh"] is not None]
    by = {}
    for key in ("direction", "class"):
        by[key] = {}
        for val in sorted({v[key] for v in valid}):
            by[key][val] = error_stats([v["error_kmh"] for v in valid if v[key] == val])
    rel = [100 * v["error_kmh"] / v["true_kmh"] for v in valid]
    out = {
        "experiment": NAME,
        "title": "Baseline accuracy",
        "pooled": pooled,
        "relative_error_pct": error_stats(rel),
        "by": by,
        "flag_counts": _flag_counts(veh),
        "jobs": job_meta(js),
    }
    write_json(out, rdir / f"{NAME}.json")
    write_vehicles_csv(results, rdir / f"{NAME}_vehicles.csv")

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1), gridspec_kw={"width_ratios": [1, 1.4]})
    e = np.array([v["error_kmh"] for v in valid])
    lim = max(2.0, float(np.ceil(np.abs(e).max() * 2) / 2)) if len(e) else 2.0
    bins = np.arange(-lim, lim + 0.25, 0.25)
    for d, c in (("towards", C_TOWARDS), ("away", C_AWAY)):
        ed = [v["error_kmh"] for v in valid if v["direction"] == d]
        ax[0].hist(ed, bins=bins, color=c, alpha=0.75, label=f"{d} (n={len(ed)})")
        vt = [v["true_kmh"] for v in valid if v["direction"] == d]
        ax[1].scatter(vt, ed, s=14, color=c, edgecolor="white", linewidth=0.5, label=d)
    ax[0].axvline(0, color=C_INK2, lw=0.8)
    ax[0].set_xlabel("speed error, measured - true (km/h)")
    ax[0].set_ylabel("vehicles")
    ax[0].legend()
    ax[1].axhline(0, color=C_INK2, lw=0.8)
    ax[1].set_xlabel("true speed (km/h)")
    ax[1].set_ylabel("speed error (km/h)")
    ax[1].legend()
    fig.suptitle("Baseline: per-vehicle speed error (simulated, 10 x 120 s)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, fdir / f"{NAME}.png")
    plt.close(fig)
    return out


def _flag_counts(veh):
    c = {}
    for v in veh:
        for f in (v["flags"] or "").split(";"):
            if f:
                c[f] = c.get(f, 0) + 1
    return dict(sorted(c.items()))
