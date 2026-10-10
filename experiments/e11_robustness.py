"""E11 - robustness to imaging conditions: shadows, noise, compression, motion blur, lighting."""

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

NAME = "e11_robustness"

CONDS = [
    ("base", "baseline (shadows, noise 2, JPEG 90)", {}),
    ("no_shadows", "no cast shadows", {"lighting.shadows": False}),
    ("low_sun", "low sun (elevation 20 deg, long shadows)", {"lighting.sun_elevation_deg": 20.0}),
    ("noise4", "sensor noise sigma 4", {"camera.noise_sigma": 4.0}),
    ("noise8", "sensor noise sigma 8", {"camera.noise_sigma": 8.0}),
    ("jpeg60", "JPEG quality 60", {"camera.jpeg_quality": 60}),
    ("jpeg30", "JPEG quality 30", {"camera.jpeg_quality": 30}),
    (
        "blur250",
        "motion blur, 1/250 s exposure",
        {"camera.exposure_s": 0.004, "camera.blur_subframes": 4},
    ),
    (
        "blur100",
        "motion blur, 1/100 s exposure",
        {"camera.exposure_s": 0.01, "camera.blur_subframes": 6},
    ),
    (
        "blur30",
        "motion blur, 1/30 s exposure",
        {"camera.exposure_s": 1 / 30, "camera.blur_subframes": 8},
    ),
    (
        "drift",
        "lighting drift +-15 % (20 s period)",
        {"lighting.drift_amplitude": 0.15, "lighting.drift_period_s": 20.0},
    ),
    (
        "jumps",
        "exposure jumps (x0.7 at 30 s, x1.4 at 60 s)",
        {"lighting.jumps": [[30.0, 0.7], [60.0, 1.4]]},
    ),
    (
        "dusk",
        "dusk: 45 % light, noise 4",
        {"lighting.gain": 0.45, "camera.noise_sigma": 4.0, "lighting.shadows": False},
    ),
]


def jobs(quick, fdir):
    seeds = [1101] if quick else [1101, 1102, 1103]
    dur = 40.0 if quick else 90.0
    conds = (
        CONDS if not quick else [c for c in CONDS if c[0] in ("base", "noise8", "blur30", "jumps")]
    )
    out = []
    for cond, label, ov in conds:
        ov = dict(ov)
        if quick and cond == "jumps":
            ov["lighting.jumps"] = [[15.0, 0.7], [28.0, 1.4]]
        out += [Job(NAME, cond, {"label": label}, s, dur, overrides=ov) for s in seeds]
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
        "title": "Robustness to imaging conditions",
        "conditions": conds,
        "jobs": job_meta(js),
    }
    write_json(out, rdir / f"{NAME}.json")
    write_vehicles_csv(results, rdir / f"{NAME}_vehicles.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 0.36 * len(conds) + 1.2), sharey=True)
    y = np.arange(len(conds))
    labels = [c["level"]["label"] for c in conds]
    p95 = [c["error_valid_kmh"].get("p95_abs", np.nan) for c in conds]
    bias = [c["error_valid_kmh"].get("bias", np.nan) for c in conds]
    ax[0].barh(y - 0.18, p95, height=0.34, color=C_TOWARDS, label="P95 |error|")
    ax[0].barh(y + 0.18, bias, height=0.34, color=C_AWAY, label="bias")
    ax[0].axvline(0, color="#52514e", lw=0.8)
    ax[0].set_yticks(y, labels, fontsize=7)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("speed error (km/h)")
    ax[0].legend(loc="lower right")
    ax[1].barh(y, [100 * c["valid_rate"] for c in conds], height=0.5, color=C_TOWARDS)
    ax[1].set_xlim(0, 100)
    ax[1].set_xlabel("vehicles with a valid speed (%)")
    fig.suptitle("Imaging conditions (simulated)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, fdir / f"{NAME}.png")
    plt.close(fig)
    return out
