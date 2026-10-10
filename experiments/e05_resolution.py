"""E05 - error vs focal length (field of view) and image resolution, zone 40-70 m.

Both change the ground sampling distance (metres of road per pixel at the zone);
the figure plots error against that common variable.
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
    write_vehicles_csv,
)

NAME = "e05_resolution"
HFOV = [30.0, 45.0, 60.0, 75.0]
RES = [(640, 360), (960, 540), (1920, 1080)]  # 1280x720 is the hfov60 condition
ZONE = {"zone.start_m": 40.0, "zone.end_m": 70.0}


def jobs(quick, fdir):
    seeds = [501] if quick else [501, 502, 503]
    dur = 40.0 if quick else 90.0
    hf = [30.0, 60.0] if quick else HFOV
    rs = [(640, 360)] if quick else RES
    out = []
    for h in hf:
        ov = {**ZONE, "camera.hfov_deg": h}
        out += [
            Job(
                NAME,
                f"hfov{h:g}",
                {"series": "field of view", "hfov_deg": h, "width": 1280},
                s,
                dur,
                overrides=ov,
            )
            for s in seeds
        ]
    for w, hh in rs:
        ov = {**ZONE, "camera.width": w, "camera.height": hh}
        out += [
            Job(
                NAME,
                f"res{w}",
                {"series": "resolution", "hfov_deg": 60.0, "width": w},
                s,
                dur,
                overrides=ov,
            )
            for s in seeds
        ]
    return out


def analyze(js, results, rdir, fdir, **_):
    plt = setup_matplotlib()
    groups = group_by_cond(results)
    conds = []
    for cond in dict.fromkeys(j.cond for j in js):
        s = pool_condition(groups[cond])
        s["cond"] = cond
        s["level"] = groups[cond][0]["level"]
        conds.append(s)
    out = {
        "experiment": NAME,
        "title": "Error vs field of view and resolution",
        "note": "zone 40-70 m, 6 m pole, 30 fps; gsd = metres of road per pixel along the road at the zone centre",
        "conditions": conds,
        "jobs": job_meta(js),
    }
    write_json(out, rdir / f"{NAME}.json")
    write_vehicles_csv(results, rdir / f"{NAME}_vehicles.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.0))
    for series, c, m in (("field of view", C_TOWARDS, "o"), ("resolution", C_AWAY, "s")):
        cs = [
            x
            for x in conds
            if x["level"]["series"] == series and x.get("error_valid_kmh", {}).get("n")
        ]
        cs.sort(key=lambda x: x["geometry"]["gsd_along_m_per_px"])
        g = [x["geometry"]["gsd_along_m_per_px"] * 100 for x in cs]
        p95 = [x["error_valid_kmh"]["p95_abs"] for x in cs]
        ci = np.array(
            [
                x["error_valid_kmh"].get("p95_abs_ci", [x["error_valid_kmh"]["p95_abs"]] * 2)
                for x in cs
            ]
        )
        yerr = [np.array(p95) - ci[:, 0], ci[:, 1] - np.array(p95)] if len(cs) else None
        ax[0].errorbar(g, p95, yerr=yerr, color=c, marker=m, ms=5, capsize=3, label=series)
        ax[1].plot(
            g, [x["error_valid_kmh"]["bias"] for x in cs], color=c, marker=m, ms=5, label=series
        )
        for x, gx, py in zip(cs, g, p95, strict=True):
            lab = (
                f"{x['level']['hfov_deg']:g} deg"
                if series == "field of view"
                else f"{x['level']['width']} px"
            )
            ax[0].annotate(
                lab, (gx, py), textcoords="offset points", xytext=(4, 4), fontsize=7, color=c
            )
    ax[0].set_xlabel("road length per pixel at the zone centre (cm)")
    ax[0].set_ylabel("P95 |speed error| (km/h)")
    ax[0].legend()
    ax[1].axhline(0, color="#52514e", lw=0.8)
    ax[1].set_xlabel("road length per pixel at the zone centre (cm)")
    ax[1].set_ylabel("bias (km/h)")
    fig.suptitle(
        "Field of view and resolution act through metres per pixel (simulated)", fontsize=10
    )
    fig.tight_layout()
    save_fig(fig, fdir / f"{NAME}.png")
    plt.close(fig)
    return out
