"""End-to-end on a short simulated video file: synth -> mp4 -> run -> records -> report."""

import json
import time

from roadspeed.cli import main


def test_synthetic_video_end_to_end(tmp_path):
    t0 = time.time()
    out = tmp_path / "sim"
    rc = main(["synth", "--out", str(out), "--preset", "small", "--seed", "1", "--duration", "15"])
    assert rc == 0
    run = tmp_path / "run"
    rc = main(
        [
            "run",
            str(out / "video.mp4"),
            "--config",
            str(out / "site.json"),
            "--out",
            str(run),
            "--quiet",
            "--review-sheets",
        ]
    )
    assert rc == 0
    assert main(["report", str(run), "--n-boot", "200"]) == 0
    gt = json.loads((out / "ground_truth.json").read_text())["vehicles"]
    recs = json.loads((run / "records.json").read_text())["records"]
    truth = [g for g in gt if g["in_window"]]
    valid = [r for r in recs if r["status"] == "valid"]
    assert len(truth) >= 3 and len(valid) >= 3
    # Every valid record matches a true vehicle (same direction, zone-centre time within
    # 0.5 s) to within 5 km/h. This is a low-resolution (640x360) smoke test of the whole
    # chain including video encoding; accuracy is characterised by the experiments.
    errs = []
    for r in valid:
        same = [
            g["speed_kmh"]
            for g in gt
            if g["direction"] == r["direction"]
            and abs(0.5 * (g["t_zone_in"] + g["t_zone_out"]) - r["t_mid_s"]) < 0.5
        ]
        assert same, r
        errs.append(min((s - r["speed_kmh"] for s in same), key=abs))
    assert max(abs(e) for e in errs) < 5.0, errs
    assert (run / "report.html").exists()
    assert time.time() - t0 < 60
