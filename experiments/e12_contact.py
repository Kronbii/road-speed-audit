"""E12 - which image point stands for the vehicle: ablation of the contact-point model.

* bottom edge, sub-pixel refined (default)
* bottom edge, raw blob boundary (no refinement)
* box centre projected with the ground homography (ignores that the point is above the road)
* box centre with the height correction from the recovered pinhole camera (assumed 0.75 m)
"""

import numpy as np

from common import (
    C_AWAY,
    C_TOWARDS,
    Job,
    group_by_cond,
    job_meta,
    pool_condition,
    save_fig,
    setup_matplotlib,
    write_json,
)

NAME = "e12_contact"
POST = [
    {"name": "centre_z0", "site": {"speed": {"contact": "center"}}},
    {"name": "centre_z0.75", "site": {"speed": {"contact": "center", "contact_height_m": 0.75}}},
]


def jobs(quick, fdir):
    seeds = [1201] if quick else [1201, 1202, 1203, 1204]
    dur = 40.0 if quick else 90.0
    out = [
        Job(
            NAME,
            "bottom_refined",
            {"model": "bottom edge, sub-pixel refined (default)"},
            s,
            dur,
            post=POST,
        )
        for s in seeds
    ]
    out += [
        Job(
            NAME,
            "bottom_raw",
            {"model": "bottom edge, raw blob boundary"},
            s,
            dur,
            site={"detector": {"backend": "bgsub", "refine_bottom": False}},
        )
        for s in seeds
    ]
    return out


def analyze(js, results, rdir, fdir, **_):
    plt = setup_matplotlib()
    groups = group_by_cond(results)
    rows = []
    for cond, key, label in (
        ("bottom_refined", "main", "bottom edge, sub-pixel refined (default)"),
        ("bottom_raw", "main", "bottom edge, raw blob boundary"),
        ("bottom_refined", "centre_z0", "box centre, ground homography"),
        ("bottom_refined", "centre_z0.75", "box centre, height-corrected (0.75 m)"),
    ):
        s = pool_condition(groups[cond], key)
        s["model"] = label
        rows.append(s)
    out = {"experiment": NAME, "title": "Contact-point model", "rows": rows, "jobs": job_meta(js)}
    write_json(out, rdir / f"{NAME}.json")
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    y = np.arange(len(rows))
    ax.barh(
        y - 0.18,
        [r["error_valid_kmh"]["p95_abs"] for r in rows],
        height=0.34,
        color=C_TOWARDS,
        label="P95 |error|",
    )
    ax.barh(
        y + 0.18,
        [r["error_valid_kmh"]["bias"] for r in rows],
        height=0.34,
        color=C_AWAY,
        label="bias",
    )
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_yticks(y, [r["model"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("speed error (km/h)")
    ax.legend(loc="lower right")
    ax.set_title("Reference point on the vehicle (simulated)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, fdir / f"{NAME}.png")
    plt.close(fig)
    return out
