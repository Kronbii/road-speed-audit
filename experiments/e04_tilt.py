"""E04 - error vs camera tilt (offset from aiming at the zone centre)."""

from common import Job, analyze_sweep

NAME = "e04_tilt"
OFFSETS = [-6.0, -3.0, 0.0, 3.0, 6.0]


def jobs(quick, fdir):
    offs = [-3.0, 0.0, 3.0] if quick else OFFSETS
    seeds = [401] if quick else [401, 402, 403]
    dur = 40.0 if quick else 90.0
    return [
        Job(
            NAME,
            f"tilt{o:+g}",
            {"tilt_offset_deg": o},
            s,
            dur,
            overrides={"camera.tilt_offset_deg": o},
        )
        for o in offs
        for s in seeds
    ]


def analyze(js, results, rdir, fdir, **_):
    return analyze_sweep(
        NAME,
        "Error vs tilt offset",
        js,
        results,
        "tilt_offset_deg",
        "tilt offset from aiming at the zone centre (deg, + = steeper)",
        rdir,
        fdir,
        note="nominal tilt aims at the zone centre (about 9.6 deg down); zone 20-50 m",
    )
