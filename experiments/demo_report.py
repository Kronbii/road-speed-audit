"""Example municipal report from a simulated clip -> results/demo/ (or results/quick/demo/).

Renders a simulated 2-lane road (speeds ~ Normal(62, 12) km/h, limit 50 km/h),
runs the same CLI chain a user would run on their own video (video file ->
roadspeed run -> roadspeed report), and keeps only the outputs a municipality would
see. The video itself is written to work/ and never committed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    from roadspeed.cli import main as cli

    work = ROOT / "work" / ("demo_quick" if args.quick else "demo")
    out = ROOT / "results" / ("quick/demo" if args.quick else "demo")
    dur = "40" if args.quick else "300"
    sets = [
        'traffic.speed_dist="normal"',
        "traffic.speed_mean_kmh=62",
        "traffic.speed_sd_kmh=12",
    ]
    argv_synth = [
        "synth",
        "--out",
        str(work / "sim"),
        "--preset",
        "baseline",
        "--seed",
        "2026",
        "--duration",
        dur,
    ]
    for s in sets:
        argv_synth += ["--set", s]
    if cli(argv_synth):
        return 1
    if cli(
        [
            "run",
            str(work / "sim" / "video.mp4"),
            "--config",
            str(work / "sim" / "site.json"),
            "--out",
            str(work / "run"),
            "--quiet",
        ]
    ):
        return 1
    if cli(["report", str(work / "run"), "--title", "Speed audit report (simulated demo clip)"]):
        return 1
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("report.html", "report.json", "records.csv", "summary.json", "review_queue.json"):
        shutil.copy(work / "run" / name, out / name)
    shutil.copytree(work / "run" / "figures", out / "figures")
    shutil.copy(work / "sim" / "ground_truth.json", out / "ground_truth.json")
    print(f"demo report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
