"""E03 - error vs camera mounting height."""

from common import Job, analyze_sweep

NAME = "e03_height"
HEIGHTS = [3.0, 4.5, 6.0, 8.0, 10.0, 12.0]


def jobs(quick, fdir):
    hs = [3.0, 6.0, 12.0] if quick else HEIGHTS
    seeds = [301] if quick else [301, 302, 303, 304]
    dur = 40.0 if quick else 90.0
    return [
        Job(NAME, f"h{h:g}", {"height_m": h}, s, dur, overrides={"camera.height_m": h})
        for h in hs
        for s in seeds
    ]


def analyze(js, results, rdir, fdir, **_):
    return analyze_sweep(
        NAME,
        "Error vs camera height",
        js,
        results,
        "height_m",
        "camera height (m)",
        rdir,
        fdir,
        note="zone 20-50 m; 60 deg HFOV, 1280x720, 30 fps",
    )
