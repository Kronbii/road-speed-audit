"""E07 - error vs true vehicle speed (analysis of the E01 baseline runs, no new runs)."""

import numpy as np

from common import C_TOWARDS, error_stats, save_fig, setup_matplotlib, write_json

NAME = "e07_speed"
BINS = [20, 40, 60, 80, 100, 120]


def jobs(quick, fdir):
    return []


def analyze(js, results, rdir, fdir, all_results=None, **_):
    plt = setup_matplotlib()
    base = (all_results or {}).get("e01_baseline", [])
    valid = [
        v
        for r in base
        if "error" not in r
        for v in r["vehicles"]
        if v["status"] == "valid" and v["error_kmh"] is not None
    ]
    rows = []
    for lo, hi in zip(BINS, BINS[1:], strict=False):
        e = [v["error_kmh"] for v in valid if lo <= v["true_kmh"] < hi]
        rel = [100 * v["error_kmh"] / v["true_kmh"] for v in valid if lo <= v["true_kmh"] < hi]
        rows.append(
            {"speed_bin_kmh": [lo, hi], "abs": error_stats(e), "relative_pct": error_stats(rel)}
        )
    out = {"experiment": NAME, "title": "Error vs true speed (from E01)", "bins": rows}
    write_json(out, rdir / f"{NAME}.json")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    data = [
        [v["error_kmh"] for v in valid if lo <= v["true_kmh"] < hi]
        for lo, hi in zip(BINS, BINS[1:], strict=False)
    ]
    pos = np.array(BINS[:-1]) + 10
    ax.boxplot(
        data,
        positions=pos,
        widths=12,
        whis=(2.5, 97.5),
        showfliers=True,
        medianprops={"color": C_TOWARDS, "linewidth": 2},
        flierprops={"markersize": 3},
    )
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(pos, [f"{lo}-{hi}" for lo, hi in zip(BINS, BINS[1:], strict=False)])
    ax.set_xlabel("true speed (km/h)")
    ax.set_ylabel("speed error (km/h)")
    ax.set_title("Error by speed band (box: quartiles, whiskers: 2.5-97.5 %)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, fdir / f"{NAME}.png")
    plt.close(fig)
    return out
