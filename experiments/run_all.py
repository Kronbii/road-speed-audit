"""Run the synthetic validation experiments and write results/ and figures/.

    python experiments/run_all.py            # full set (roughly 1.5-2 h with 6 worker processes)
    python experiments/run_all.py --quick    # small version -> results/quick, figures/quick
    python experiments/run_all.py --only e03_height e06_fps

All scenes are simulated. Seeds are fixed; outputs are byte-identical between runs
on the same software stack.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import ROOT, run_jobs, workers_default, write_json  # noqa: E402

EXPERIMENTS = [
    "e01_baseline",
    "e02_distance",
    "e03_height",
    "e04_tilt",
    "e05_resolution",
    "e06_fps",
    "e07_speed",
    "e08_density",
    "e09_calibration",
    "e10_v85",
    "e11_robustness",
    "e12_contact",
    "e13_probe",
]
DEPENDS = {"e07_speed": ["e01_baseline"], "e13_probe": ["e01_baseline", "e05_resolution"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=workers_default())
    args = ap.parse_args(argv)

    rdir = ROOT / "results" / ("quick" if args.quick else "")
    fdir = ROOT / "figures" / ("quick" if args.quick else "")
    names = args.only or EXPERIMENTS
    for n in list(names):
        for dep in DEPENDS.get(n, []):
            if dep not in names:
                names.insert(0, dep)
    names = [n for n in EXPERIMENTS if n in names]
    mods = {n: importlib.import_module(n) for n in names}
    all_jobs = {n: m.jobs(args.quick, fdir) for n, m in mods.items()}
    flat = [j for n in names for j in all_jobs[n]]
    print(
        f"{len(flat)} simulated runs across {len(names)} experiments, {args.workers} workers",
        flush=True,
    )
    t0 = time.time()
    res = run_jobs(flat, args.workers)
    print(f"runs finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    by_exp: dict[str, list] = {n: [] for n in names}
    for r in res:
        by_exp[r["exp"]].append(r)
    for n in names:
        mods[n].analyze(all_jobs[n], by_exp[n], rdir, fdir, all_results=by_exp)
        print(f"  analysed {n}", flush=True)
    import cv2
    import matplotlib
    import numpy
    import scipy

    import roadspeed

    write_json(
        {
            "experiments": names,
            "quick": args.quick,
            "software": {
                "roadspeed": roadspeed.__version__,
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "opencv": cv2.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "note": "all results are from simulated video; see README for the simulation's limits",
        },
        rdir / "manifest.json",
    )
    if not args.only:
        from guidance import write_guidance

        write_guidance(rdir)
        from tables import write_tables

        write_tables(rdir)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {rdir} {fdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
