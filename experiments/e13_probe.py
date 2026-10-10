"""E13 - can a few reference passes remove the residual bias? (analysis only)

Simulates the field-protocol idea: k vehicles with independently known speed (a
probe car with GPS, or a radar reading) give a site correction factor
median(true / measured), applied to all other vehicles of the same run. Uses the
valid vehicles of E01 (720p) and the 640x360 condition of E05.
"""

import numpy as np

from common import C_AWAY, C_TOWARDS, save_fig, setup_matplotlib, write_json

NAME = "e13_probe"
KS = [1, 3, 5, 10, 20]
DRAWS = 500


def jobs(quick, fdir):
    return []


def _runs(results, cond=None):
    out = []
    for r in results:
        if "error" in r or (cond and r["cond"] != cond):
            continue
        v = [
            (x["true_kmh"], x["meas_kmh"])
            for x in r["vehicles"]
            if x["status"] == "valid" and x["meas_kmh"] is not None
        ]
        if len(v) >= 6:
            out.append(np.array(v))
    return out


def _study(runs, seed):
    rng = np.random.default_rng(seed)
    res = {"uncorrected": None, "k": {}}
    before = [100 * np.mean(r[:, 1] / r[:, 0] - 1) for r in runs]
    res["uncorrected"] = {
        "run_bias_pct_mean": float(np.mean(before)),
        "run_bias_abs_pct_mean": float(np.mean(np.abs(before))),
    }
    for k in KS:
        after = []
        for r in runs:
            if len(r) <= k + 2:
                continue
            for _ in range(DRAWS):
                idx = rng.permutation(len(r))
                probe, rest = r[idx[:k]], r[idx[k:]]
                f = np.median(probe[:, 0] / probe[:, 1])
                after.append(100 * np.mean(rest[:, 1] * f / rest[:, 0] - 1))
        a = np.array(after)
        res["k"][str(k)] = {
            "run_bias_pct_mean": float(a.mean()) if len(a) else None,
            "run_bias_abs_pct_mean": float(np.abs(a).mean()) if len(a) else None,
            "run_bias_abs_pct_p95": float(np.percentile(np.abs(a), 95)) if len(a) else None,
        }
    return res


def analyze(js, results, rdir, fdir, all_results=None, **_):
    plt = setup_matplotlib()
    allr = all_results or {}
    cases = {
        "720p_baseline": _runs(allr.get("e01_baseline", [])),
        "360p_zone40-70": _runs(allr.get("e05_resolution", []), "res640"),
    }
    out = {
        "experiment": NAME,
        "title": "Correction from k reference passes",
        "draws_per_run": DRAWS,
        "cases": {},
    }
    for i, (name, runs) in enumerate(cases.items()):
        if runs:
            out["cases"][name] = {"runs": len(runs), **_study(runs, 13 + i)}
    write_json(out, rdir / f"{NAME}.json")
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for (name, c), col in zip(out["cases"].items(), (C_TOWARDS, C_AWAY), strict=False):
        xs = [0] + KS
        ys = [c["uncorrected"]["run_bias_abs_pct_mean"]] + [
            c["k"][str(k)]["run_bias_abs_pct_mean"] for k in KS
        ]
        ax.plot(xs, ys, color=col, marker="o", ms=5, label=name)
    ax.set_xlabel("reference passes used for the correction (0 = none)")
    ax.set_ylabel("mean |run bias| (% of speed)")
    if out["cases"]:
        ax.legend()
    ax.set_title("Site correction from known-speed passes (simulated)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, fdir / f"{NAME}.png")
    plt.close(fig)
    return out
