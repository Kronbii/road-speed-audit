"""E10 - municipal statistics on synthetic traffic: mean speed, V85, share over the limit, flow.

Speeds ~ Normal(62, 12) km/h truncated to 20-120 (an assumed distribution, not a
measurement), limit 50 km/h. Each run's report (with 95 % intervals) is compared
with the truth of the simulated vehicles that crossed the zone.
"""

import numpy as np

from common import C_TOWARDS, Job, job_meta, save_fig, setup_matplotlib, write_json
from roadspeed.report import summarize
from roadspeed.synth import ScenarioConfig

NAME = "e10_v85"
LIMIT = 50.0
WARMUP = ScenarioConfig().warmup_s
OV = {"traffic.speed_dist": "normal", "traffic.speed_mean_kmh": 62.0, "traffic.speed_sd_kmh": 12.0}


def jobs(quick, fdir):
    seeds = [1001, 1002, 1003] if quick else list(range(1001, 1021))
    dur = 60.0 if quick else 180.0
    return [
        Job(
            NAME,
            "normal62",
            {},
            s,
            dur,
            overrides=OV,
            site={"speed_limit_kmh": LIMIT},
            keep_records=True,
        )
        for s in seeds
    ]


def _true_stats(v):
    v = np.asarray(v, float)
    return {
        "mean_kmh": float(v.mean()),
        "v85_kmh": float(np.percentile(v, 85)),
        "share_over_limit": float(np.mean(v > LIMIT)),
    }


def analyze(js, results, rdir, fdir, **_):
    plt = setup_matplotlib()
    runs = []
    for r in results:
        if "error" in r:
            continue
        truth = [v["true_kmh"] for v in r["vehicles"]]  # vehicles crossing the zone in the window
        ts = _true_stats(truth)
        # restrict records to the same scored window as the ground truth
        warm, dur = WARMUP, r["run_summary"]["duration_s"]
        recs = [
            x
            for x in r["records"]
            if x["t_zone_in_s"] is not None
            and x["t_zone_in_s"] >= warm
            and x["t_zone_out_s"] <= dur
        ]
        summ_w = {**r["run_summary"], "duration_s": dur - warm}
        rep = summarize(summ_w, recs, {}, n_boot=1000, seed=r["seed"])
        g = rep["groups"]["all"]
        row = {
            "seed": r["seed"],
            "n_true": len(truth),
            "n_measured": g["n"],
            "n_counted": g["n_counted"],
        }
        for k in ("mean_kmh", "v85_kmh", "share_over_limit"):
            ci = g[k + "_ci"]
            row[k] = {
                "true": ts[k],
                "measured": g[k],
                "error": g[k] - ts[k],
                "ci": ci,
                "covered": bool(ci[0] is not None and ci[0] <= ts[k] <= ci[1]),
            }
        # flow: counted records vs simulated vehicles crossing the zone in the scored window
        dur_h = (dur - warm) / 3600.0
        row["flow_veh_h"] = {
            "true": len(truth) / dur_h,
            "measured": g.get("flow_veh_h"),
            "ci": g.get("flow_veh_h_ci"),
        }
        runs.append(row)
    summ = {}
    for k in ("mean_kmh", "v85_kmh", "share_over_limit"):
        e = np.array([x[k]["error"] for x in runs])
        summ[k] = {
            "mean_error": float(e.mean()),
            "rmse": float(np.sqrt((e**2).mean())),
            "max_abs_error": float(np.abs(e).max()),
            "ci_coverage": float(np.mean([x[k]["covered"] for x in runs])),
            "mean_ci_width": float(np.mean([x[k]["ci"][1] - x[k]["ci"][0] for x in runs])),
            "runs": len(runs),
        }
    fe = np.array([x["flow_veh_h"]["measured"] - x["flow_veh_h"]["true"] for x in runs])
    summ["flow_veh_h"] = {
        "mean_error": float(fe.mean()),
        "mean_relative_error_pct": float(
            np.mean(
                [100 * (x["flow_veh_h"]["measured"] / x["flow_veh_h"]["true"] - 1) for x in runs]
            )
        ),
        "count_error_mean": float(np.mean([x["n_counted"] - x["n_true"] for x in runs])),
        "count_error_abs_max": int(max(abs(x["n_counted"] - x["n_true"]) for x in runs)),
        "note": "the flow interval describes the rate behind the count; here the realised count is compared",
    }
    out = {
        "experiment": NAME,
        "title": "Mean speed, V85 and share over the limit on synthetic traffic",
        "note": "speeds ~ Normal(62, 12) truncated 20-120 km/h (assumed), limit 50 km/h, 2-lane road, 300 veh/h/lane",
        "summary": summ,
        "runs": runs,
        "jobs": job_meta(js),
    }
    write_json(out, rdir / f"{NAME}.json")
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))
    order = sorted(runs, key=lambda x: x["v85_kmh"]["true"])
    y = np.arange(len(order))
    for i, x in enumerate(order):
        ci = x["v85_kmh"]["ci"]
        ax[0].plot(ci, [i, i], color=C_TOWARDS, lw=2, solid_capstyle="round")
        ax[0].plot(x["v85_kmh"]["measured"], i, "o", color=C_TOWARDS, ms=4)
        ax[0].plot(x["v85_kmh"]["true"], i, "|", color="#0b0b0b", ms=9, mew=1.6)
    ax[0].set_yticks(y, [str(x["seed"]) for x in order], fontsize=6)
    ax[0].set_xlabel("V85 (km/h): measured with 95 % CI, true value as |")
    ax[0].set_ylabel("simulated run (seed)")
    e = [x["v85_kmh"]["error"] for x in runs]
    ax[1].hist(e, bins=np.arange(-3, 3.25, 0.25), color=C_TOWARDS, alpha=0.8)
    ax[1].axvline(0, color="#52514e", lw=0.8)
    ax[1].set_xlabel("V85 error, measured - true (km/h)")
    ax[1].set_ylabel("runs")
    fig.suptitle(f"V85 on synthetic traffic: {len(runs)} runs (simulated)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, fdir / f"{NAME}.png")
    plt.close(fig)
    return out
