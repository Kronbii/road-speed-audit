"""E08 - traffic density, occlusion and tracker ablations: tracking quality and speed error."""

import numpy as np

from common import (
    C_AWAY,
    C_THIRD,
    C_TOWARDS,
    Job,
    group_by_cond,
    job_meta,
    pool_condition,
    save_fig,
    setup_matplotlib,
    write_json,
    write_vehicles_csv,
)

NAME = "e08_density"
POLE = [{"x": 4.06, "y": 8.75, "w": 0.35, "d": 0.35, "h": 8.0}]  # lamp post near the camera
NO_ROBUST = {"max_lost_s": 0.1, "handle_merges": False}


def jobs(quick, fdir):
    seeds = [801] if quick else [801, 802, 803]
    dur = 40.0 if quick else 90.0
    conds = []
    flows2 = [150, 500] if quick else [150, 300, 500, 700]
    flows4 = [450] if quick else [300, 450, 600]
    for f in flows2:
        conds.append(
            (
                f"2lane_f{f}",
                {"road": "2-lane", "flow": f, "variant": "default"},
                {"traffic.flow_veh_h_per_lane": float(f)},
                {},
            )
        )
    for f in flows4:
        conds.append(
            (
                f"4lane_f{f}",
                {"road": "4-lane", "flow": f, "variant": "default"},
                {"road.lanes_per_direction": 2, "traffic.flow_veh_h_per_lane": float(f)},
                {},
            )
        )
    conds.append(
        (
            "2lane_pole",
            {"road": "2-lane", "flow": 300, "variant": "lamp post"},
            {"occluders": POLE},
            {},
        )
    )
    conds.append(
        (
            "2lane_pole_noocc",
            {"road": "2-lane", "flow": 300, "variant": "lamp post, no occlusion handling"},
            {"occluders": POLE},
            {"tracker": NO_ROBUST},
        )
    )
    conds.append(
        (
            "4lane_f450_noocc",
            {"road": "4-lane", "flow": 450, "variant": "no occlusion handling"},
            {"road.lanes_per_direction": 2, "traffic.flow_veh_h_per_lane": 450.0},
            {"tracker": NO_ROBUST},
        )
    )
    if quick:
        conds = [c for c in conds if c[0] != "4lane_f450_noocc"]
    return [
        Job(NAME, c, lvl, s, dur, overrides=ov, site=site)
        for c, lvl, ov, site in conds
        for s in seeds
    ]


def analyze(js, results, rdir, fdir, **_):
    plt = setup_matplotlib()
    groups = group_by_cond(results)
    conds = []
    for cond in dict.fromkeys(j.cond for j in js):
        s = pool_condition(groups[cond])
        s["cond"] = cond
        s["level"] = groups[cond][0]["level"]
        n = max(s["gt_vehicles"], 1)
        t = s["tracking"]
        s["zone_id_switches_per_100_veh"] = 100 * t["zone_id_switches"] / n
        s["zone_fragmentations_per_100_veh"] = 100 * t["zone_fragmentations"] / n
        conds.append(s)
    out = {
        "experiment": NAME,
        "title": "Traffic density, occlusion and tracker ablations",
        "note": "flow per lane in veh/h; 'no occlusion handling' = tracks dropped after 0.1 s and no merged-blob logic",
        "conditions": conds,
        "jobs": job_meta(js),
    }
    write_json(out, rdir / f"{NAME}.json")
    write_vehicles_csv(results, rdir / f"{NAME}_vehicles.csv")
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for road, c, m in (("2-lane", C_TOWARDS, "o"), ("4-lane", C_AWAY, "s")):
        cs = sorted(
            [x for x in conds if x["level"]["road"] == road and x["level"]["variant"] == "default"],
            key=lambda x: x["level"]["flow"],
        )
        fl = [x["level"]["flow"] for x in cs]
        ax[0].plot(fl, [100 * x["valid_rate"] for x in cs], color=c, marker=m, ms=5, label=road)
        ax[1].plot(
            fl,
            [x["zone_id_switches_per_100_veh"] + x["zone_fragmentations_per_100_veh"] for x in cs],
            color=c,
            marker=m,
            ms=5,
            label=road,
        )
        ax[2].plot(
            fl,
            [x["error_valid_kmh"].get("p95_abs", np.nan) for x in cs],
            color=c,
            marker=m,
            ms=5,
            label=road,
        )
    ax[0].set_ylabel("vehicles with a valid speed (%)")
    ax[1].set_ylabel("ID switches + fragmentations\nin the zone per 100 vehicles")
    ax[2].set_ylabel("P95 |speed error| (km/h)")
    for a in ax:
        a.set_xlabel("flow per lane (veh/h)")
        a.legend()
    ax[0].set_ylim(0, 105)
    fig.suptitle(
        "Denser traffic means more merged blobs and fewer valid speeds (simulated)", fontsize=10
    )
    fig.tight_layout()
    save_fig(fig, fdir / f"{NAME}.png")
    plt.close(fig)
    # ablation bars
    abl = [
        x
        for x in conds
        if x["cond"]
        in ("2lane_f300", "2lane_pole", "2lane_pole_noocc", "4lane_f450", "4lane_f450_noocc")
    ]
    if abl:
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        labels = [
            f"{x['level']['road']}, {x['level']['flow']}/h\n{x['level']['variant']}" for x in abl
        ]
        y = np.arange(len(abl))
        ax.barh(
            y,
            [100 * x["valid_rate"] for x in abl],
            color=[C_THIRD if "no occ" in x["level"]["variant"] else C_TOWARDS for x in abl],
            height=0.6,
        )
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_xlabel("vehicles with a valid speed (%)")
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        fig.tight_layout()
        save_fig(fig, fdir / f"{NAME}_ablation.png")
        plt.close(fig)
    return out
