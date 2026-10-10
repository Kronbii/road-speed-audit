"""Turn the sweep results into placement guidance, stating only what the results support.

A condition "meets the target" when, pooled over its simulated runs, the 95th
percentile of |speed error| of valid records is <= TARGET_P95 km/h and at least
MIN_VALID of the simulated vehicles crossing the zone got a valid speed. The
guidance is the range of each swept parameter over which every tested level meets
the target (one parameter varied at a time from the baseline set-up).
"""

from __future__ import annotations

import json
from pathlib import Path

TARGETS = [(2.0, 0.8), (1.0, 0.8)]  # (P95 |error| km/h, minimum valid share)


def _load(rdir: Path, name: str):
    p = rdir / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _meets(c, p95, vr):
    e = c.get("error_valid_kmh") or {}
    return bool(e.get("n", 0) >= 10 and e["p95_abs"] <= p95 and c.get("valid_rate", 0) >= vr)


def _range(conds, key, p95, vr, direction):
    """Largest contiguous range from the baseline end where all levels meet the target."""
    pts = sorted(
        ((c["level"][key], _meets(c, p95, vr)) for c in conds if key in c.get("level", {})),
        key=lambda x: x[0],
    )
    if direction == "min":  # want the smallest value from which all larger values pass
        ok = None
        for x, m in reversed(pts):
            if not m:
                break
            ok = x
        return ok, pts
    ok = None
    for x, m in pts:
        if not m:
            break
        ok = x
    return ok, pts


def write_guidance(rdir: Path) -> dict:
    out = {
        "criteria": [],
        "statements": [],
        "caveat": (
            "Simulation only: flat straight road, ideal pinhole lens, box-shaped vehicles, background "
            "subtraction detector, one parameter varied at a time around a 6 m / 60 deg / 1280x720 / 30 fps "
            "baseline with the zone 20-50 m from the pole. Treat as a starting point for the field pilot, not "
            "as validated limits."
        ),
    }
    sweeps = [
        (
            "e03_height",
            "height_m",
            "min",
            "camera height (m)",
            "mount the camera at least {v:g} m high",
        ),
        (
            "e02_distance",
            "zone_centre_m",
            "max",
            "zone centre distance (m)",
            "keep the zone centre within {v:g} m of the pole",
        ),
        ("e06_fps", "fps", "min", "frame rate", "record at {v:g} frames/s or more"),
        ("e04_tilt", "tilt_offset_deg", "max", "tilt offset (deg)", None),
    ]
    for p95, vr in TARGETS:
        crit = f"P95 |error| <= {p95:g} km/h and >= {vr:.0%} of vehicles measured"
        out["criteria"].append(crit)
        for name, key, direction, label, text in sweeps:
            d = _load(rdir, name)
            if not d:
                continue
            v, pts = _range(d["conditions"], key, p95, vr, direction)
            st = {
                "criterion": crit,
                "parameter": label,
                "tested": [x for x, _ in pts],
                "meets": [m for _, m in pts],
                "source": f"results/{name}.json",
            }
            if v is not None and text:
                st["statement"] = (
                    text.format(v=v)
                    + " (tested levels: "
                    + ", ".join(f"{x:g}" for x, _ in pts)
                    + ")"
                )
            elif text:
                st["statement"] = f"no tested {label} level met the criterion from the baseline end"
            out["statements"].append(st)
        d = _load(rdir, "e05_resolution")
        if d:
            pts = sorted(
                (
                    (c["geometry"]["gsd_along_m_per_px"], _meets(c, p95, vr), c["cond"])
                    for c in d["conditions"]
                    if c.get("geometry")
                ),
                key=lambda x: x[0],
            )
            ok = None
            for g, m, _ in pts:
                if not m:
                    break
                ok = g
            out["statements"].append(
                {
                    "criterion": crit,
                    "parameter": "road length per pixel at the zone centre (m)",
                    "tested": [round(g, 4) for g, _, _ in pts],
                    "conditions": [c for _, _, c in pts],
                    "meets": [m for _, m, _ in pts],
                    "statement": (
                        f"choose lens and resolution so one pixel covers at most {100 * ok:.0f} cm of road (along the road) at the zone centre"
                        if ok is not None
                        else "no tested lens/resolution met the criterion"
                    ),
                    "source": "results/e05_resolution.json",
                }
            )
        d = _load(rdir, "e08_density")
        if d:
            for road in ("2-lane", "4-lane"):
                cs = sorted(
                    [
                        c
                        for c in d["conditions"]
                        if c["level"]["road"] == road and c["level"]["variant"] == "default"
                    ],
                    key=lambda c: c["level"]["flow"],
                )
                ok = None
                for c in cs:
                    if not _meets(c, p95, vr):
                        break
                    ok = c["level"]["flow"]
                out["statements"].append(
                    {
                        "criterion": crit,
                        "parameter": f"flow per lane, {road} (veh/h)",
                        "tested": [c["level"]["flow"] for c in cs],
                        "meets": [_meets(c, p95, vr) for c in cs],
                        "statement": (
                            f"{road}: target met up to {ok} veh/h per lane"
                            if ok
                            else f"{road}: target not met at the lowest tested flow"
                        ),
                        "source": "results/e08_density.json",
                    }
                )
    from common import write_json

    write_json(out, rdir / "guidance.json")
    return out
