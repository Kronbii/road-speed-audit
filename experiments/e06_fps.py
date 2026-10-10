"""E06 - error vs frame rate."""

from common import Job, analyze_sweep

NAME = "e06_fps"
FPS = [10.0, 15.0, 20.0, 25.0, 30.0, 50.0]


def jobs(quick, fdir):
    fs = [10.0, 30.0] if quick else FPS
    seeds = [601] if quick else [601, 602, 603, 604]
    dur = 40.0 if quick else 90.0
    return [
        Job(NAME, f"fps{f:g}", {"fps": f}, s, dur, overrides={"camera.fps": f})
        for f in fs
        for s in seeds
    ]


def analyze(js, results, rdir, fdir, **_):
    return analyze_sweep(
        NAME,
        "Error vs frame rate",
        js,
        results,
        "fps",
        "frame rate (frames/s)",
        rdir,
        fdir,
        note="6 m pole, zone 20-50 m, 1280x720",
    )
