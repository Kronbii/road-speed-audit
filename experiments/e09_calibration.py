"""E09 - sensitivity to calibration error: click noise, survey (tape) noise, point layout.

Detection and tracking are run once per seed; speed estimation is then repeated with
many independently simulated field calibrations. Calibration error is shared by
every vehicle of a run, so the key quantity is the per-run bias it induces.
"""

import numpy as np

from common import (
    C_AWAY,
    C_TOWARDS,
    Job,
    error_stats,
    job_meta,
    save_fig,
    setup_matplotlib,
    write_json,
)

NAME = "e09_calibration"
CLICK = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
SURVEY = [0.0, 0.02, 0.05, 0.10]
LAYOUT = ["compact", "near_half", "zone", "wide"]
UNC = {"n_samples": 60}


def _variants(quick):
    reps = 3 if quick else 8
    click = [0.5, 2.0] if quick else CLICK
    survey = [0.0, 0.05] if quick else SURVEY
    layout = ["compact", "zone"] if quick else LAYOUT
    v = []
    for c in click:
        for r in range(reps):
            v.append(
                {
                    "name": f"click={c:g}/rep={r}",
                    "site": {
                        "click_sigma_px": c,
                        "ground_sigma_m": 0.0,
                        "seed": 100 + r,
                        "uncertainty": UNC,
                    },
                }
            )
    for g in survey:
        for r in range(reps):
            v.append(
                {
                    "name": f"survey={g:g}/rep={r}",
                    "site": {
                        "click_sigma_px": 1.0,
                        "ground_sigma_m": g,
                        "seed": 200 + r,
                        "uncertainty": UNC,
                    },
                }
            )
    for lay in layout:
        for r in range(reps):
            v.append(
                {
                    "name": f"layout={lay}/rep={r}",
                    "site": {
                        "click_sigma_px": 1.0,
                        "ground_sigma_m": 0.02,
                        "layout": lay,
                        "seed": 300 + r,
                        "uncertainty": UNC,
                    },
                }
            )
    return v


def jobs(quick, fdir):
    seeds = [901] if quick else [901, 902, 903, 904, 905, 906]
    dur = 45.0 if quick else 90.0
    return [Job(NAME, "base", {}, s, dur, post=_variants(quick)) for s in seeds]


def analyze(js, results, rdir, fdir, **_):
    plt = setup_matplotlib()
    fam: dict[str, dict] = {}
    for r in results:
        if "error" in r:
            continue
        for name, p in r["post"].items():
            if "metrics" not in p:
                continue
            family, rep = name.split("/")
            errs = [
                v["error_kmh"]
                for v in p["vehicles"]
                if v["status"] == "valid" and v["error_kmh"] is not None
            ]
            rel = [
                100 * v["error_kmh"] / v["true_kmh"]
                for v in p["vehicles"]
                if v["status"] == "valid" and v["error_kmh"] is not None
            ]
            cov = [
                abs(v["error_kmh"]) <= 1.96 * v["u_kmh"]
                for v in p["vehicles"]
                if v["status"] == "valid" and v["error_kmh"] is not None and v["u_kmh"]
            ]
            d = fam.setdefault(
                family, {"run_bias": [], "run_bias_pct": [], "errors": [], "coverage": [], "u": []}
            )
            if errs:
                d["run_bias"].append(float(np.mean(errs)))
                d["run_bias_pct"].append(float(np.mean(rel)))
            d["errors"] += errs
            d["coverage"] += cov
            d["u"] += [v["u_kmh"] for v in p["vehicles"] if v["status"] == "valid" and v["u_kmh"]]
    rows = []
    for family, d in fam.items():
        kind, val = family.split("=")
        rb = np.abs(d["run_bias_pct"])
        rows.append(
            {
                "family": kind,
                "value": float(val) if kind != "layout" else val,
                "calibrations": len(d["run_bias"]),
                "run_bias_abs_pct_mean": float(rb.mean()) if len(rb) else None,
                "run_bias_abs_pct_p95": float(np.percentile(rb, 95)) if len(rb) else None,
                "run_bias_kmh_sd": float(np.std(d["run_bias"])) if d["run_bias"] else None,
                "pooled_error_kmh": error_stats(d["errors"]),
                "ci95_coverage": float(np.mean(d["coverage"])) if d["coverage"] else None,
                "median_u_kmh": float(np.median(d["u"])) if d["u"] else None,
            }
        )
    order = {"click": 0, "survey": 1, "layout": 2}
    lay_order = {k: i for i, k in enumerate(LAYOUT)}
    rows.sort(
        key=lambda x: (
            order[x["family"]],
            lay_order.get(x["value"], 0) if x["family"] == "layout" else x["value"],
        )
    )
    out = {
        "experiment": NAME,
        "title": "Sensitivity to calibration error",
        "note": (
            "run bias = mean speed error of the valid vehicles of one run under one simulated calibration; "
            "click noise in px per axis (clicks rounded to whole pixels), survey noise in m per axis; "
            "layouts: compact = 4 points within 6 m at the zone start, near_half = 9 points over the near half, "
            "zone = 9 points at zone start/middle/end, wide = 9 points 10 m beyond both ends"
        ),
        "rows": rows,
        "jobs": job_meta(js),
    }
    write_json(out, rdir / f"{NAME}.json")
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for a, kind, xl in (
        (ax[0], "click", "click noise per axis (px)"),
        (ax[1], "survey", "survey noise per axis (m)"),
    ):
        rs = [x for x in rows if x["family"] == kind]
        xs = [x["value"] for x in rs]
        a.plot(
            xs,
            [x["run_bias_abs_pct_p95"] for x in rs],
            color=C_TOWARDS,
            marker="o",
            ms=5,
            label="P95 over calibrations",
        )
        a.plot(
            xs,
            [x["run_bias_abs_pct_mean"] for x in rs],
            color=C_AWAY,
            marker="s",
            ms=5,
            label="mean over calibrations",
        )
        a.set_xlabel(xl)
        a.set_ylabel("|calibration-induced bias| (% of speed)")
        a.legend()
    rs = [x for x in rows if x["family"] == "layout"]
    y = np.arange(len(rs))
    ax[2].barh(y, [x["run_bias_abs_pct_p95"] for x in rs], color=C_TOWARDS, height=0.6)
    ax[2].set_yticks(y, [x["value"] for x in rs])
    ax[2].invert_yaxis()
    ax[2].set_xlabel("P95 |bias| (% of speed), 1 px clicks")
    ax[2].set_title("calibration point layout", fontsize=9)
    fig.suptitle(
        "Calibration error shifts every speed of a run by the same few per cent (simulated)",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, fdir / f"{NAME}.png")
    plt.close(fig)
    return out
