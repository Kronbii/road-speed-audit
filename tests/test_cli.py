import json

import numpy as np

from roadspeed.cli import main


def test_calibrate_from_points(tmp_path, camera):
    g = np.array([[x, y] for y in (20.0, 35.0, 50.0) for x in (-3.5, 0.0, 3.5)])
    uv, _ = camera.project(np.column_stack([g, np.zeros(len(g))]))
    pairs = [{"image": u.tolist(), "ground": p.tolist()} for u, p in zip(uv, g, strict=True)]
    (tmp_path / "pairs.json").write_text(json.dumps(pairs))
    out = tmp_path / "site.json"
    rc = main(
        [
            "calibrate",
            "--points",
            str(tmp_path / "pairs.json"),
            "--size",
            "1280",
            "720",
            "--out",
            str(out),
            "--zone",
            "20",
            "50",
            "--lane",
            "t:-1.75:3.5:towards",
            "--lane",
            "a:1.75:3.5:away",
            "--speed-limit",
            "50",
        ]
    )
    assert rc == 0
    site = json.loads(out.read_text())
    assert site["zone"] == {"start_m": 20.0, "end_m": 50.0}
    assert len(site["lanes"]) == 2 and site["speed_limit_kmh"] == 50
