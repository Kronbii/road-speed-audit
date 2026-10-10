"""E02 - error vs distance of the measurement zone from the camera (zone length 30 m)."""

from common import Job, analyze_sweep

NAME = "e02_distance"
STARTS = [10, 25, 40, 60, 80, 100]


def jobs(quick, fdir):
    starts = [10, 40, 80] if quick else STARTS
    seeds = [201] if quick else [201, 202, 203, 204]
    dur = 40.0 if quick else 90.0
    out = []
    for z in starts:
        lvl = {"zone_start_m": z, "zone_centre_m": z + 15}
        ov = {"zone.start_m": float(z), "zone.end_m": float(z + 30)}
        out += [Job(NAME, f"zone{z}", lvl, s, dur, overrides=ov) for s in seeds]
    return out


def analyze(js, results, rdir, fdir, **_):
    return analyze_sweep(
        NAME,
        "Error vs distance of the zone from the camera",
        js,
        results,
        "zone_centre_m",
        "distance of zone centre from the pole (m)",
        rdir,
        fdir,
        note="30 m zone; camera re-aimed at the zone centre; 6 m pole, 60 deg HFOV, 1280x720, 30 fps",
    )
