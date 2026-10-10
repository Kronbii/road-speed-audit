"""Per-vehicle records and their CSV / JSON serialisation.

Records hold numbers only: no images, no plate or face data. Floats are rounded
on output so that runs from the same input and seed are byte-identical.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

RECORD_FIELDS = [
    "record_id",
    "track_id",
    "segment",
    "t_zone_in_s",
    "t_zone_out_s",
    "t_mid_s",
    "wall_time",
    "lane",
    "direction",
    "speed_kmh",
    "u_kmh",
    "u_cal_kmh",
    "u_rand_kmh",
    "ci95_low_kmh",
    "ci95_high_kmh",
    "n_points",
    "coverage",
    "residual_m",
    "width_m",
    "lateral_m",
    "pixel_sigma_px",
    "label",
    "status",
    "flags",
]


@dataclass
class VehicleRecord:
    record_id: int
    track_id: int
    segment: int
    t_zone_in_s: float | None
    t_zone_out_s: float | None
    t_mid_s: float | None
    wall_time: str | None
    lane: str | None
    direction: str | None
    speed_kmh: float | None
    u_kmh: float | None
    u_cal_kmh: float | None
    u_rand_kmh: float | None
    ci95_low_kmh: float | None
    ci95_high_kmh: float | None
    n_points: int
    coverage: float
    residual_m: float | None
    width_m: float | None
    lateral_m: float | None
    pixel_sigma_px: float | None
    label: str
    status: str
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = _r(v, k)
        return d


def _r(v: float | None, key: str = ""):
    if v is None:
        return None
    if v != v:  # NaN
        return None
    nd = 3 if key.endswith("_s") else 4
    return round(float(v), nd)


def write_records_csv(records: list[VehicleRecord], path) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=RECORD_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in records:
        d = r.as_dict()
        d["flags"] = ";".join(r.flags)
        w.writerow({k: ("" if d[k] is None else d[k]) for k in RECORD_FIELDS})
    Path(path).write_text(buf.getvalue())


def read_records_csv(path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out = {}
            for k, v in row.items():
                if v == "":
                    out[k] = None
                elif k in ("record_id", "track_id", "segment", "n_points"):
                    out[k] = int(v)
                elif k in ("lane", "direction", "label", "status", "wall_time"):
                    out[k] = v
                elif k == "flags":
                    out[k] = v.split(";") if v else []
                else:
                    out[k] = float(v)
            if "flags" not in out or out["flags"] is None:
                out["flags"] = []
            rows.append(out)
    return rows


def dump_json(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
