"""Command-line interface: ``roadspeed calibrate | run | report | review | synth``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _lane_arg(s: str) -> dict:
    # name:center_m:width_m:direction
    parts = s.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("lane must be name:center_m:width_m:direction")
    name, c, w, d = parts
    if d not in ("away", "towards", "both"):
        raise argparse.ArgumentTypeError("lane direction must be away, towards or both")
    return {"name": name, "center_m": float(c), "width_m": float(w), "direction": d}


def _read_frame(args):
    import cv2

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"cannot read image {args.image}")
        return img
    if args.video:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"cannot read frame {args.frame} of {args.video}")
        return img
    return None


def cmd_calibrate(args) -> int:
    from .calibration import (
        Calibration,
        SiteConfig,
        draw_overlay,
        pick_points,
        points_from_lane_markings,
    )

    frame = _read_frame(args)
    labels = None
    if args.points:
        pairs = json.loads(Path(args.points).read_text())
        img = np.array([p["image"] for p in pairs], float)
        gnd = np.array([p["ground"] for p in pairs], float)
    elif args.lanes:
        spec = json.loads(Path(args.lanes).read_text())
        img, gnd = points_from_lane_markings(
            spec["lines"], spec["dash_length_m"], spec["dash_gap_m"]
        )
    else:
        if frame is None:
            sys.exit("interactive calibration needs --image or --video")
        if args.survey:
            survey = json.loads(Path(args.survey).read_text())
            labels = [p["label"] for p in survey]
            gnd = np.array([p["ground"] for p in survey], float)
            img = pick_points(frame, labels)
        else:
            img = pick_points(frame)
            gnd = []
            for i, (u, v) in enumerate(img):
                txt = input(f"ground X Y (metres) for point {i + 1} at pixel ({u:.0f}, {v:.0f}): ")
                x, y = (float(t) for t in txt.replace(",", " ").split())
                gnd.append([x, y])
            gnd = np.array(gnd)
    if frame is not None:
        size = [int(frame.shape[1]), int(frame.shape[0])]
    elif args.size:
        size = list(args.size)
    else:
        sys.exit("give --image/--video or --size W H so the image size is known")
    cal = Calibration(
        image_points=np.round(img, 3).tolist(),
        ground_points=np.round(gnd, 4).tolist(),
        image_size=size,
        click_sigma_px=args.click_sigma,
        ground_sigma_m=args.ground_sigma,
        focal_px=args.focal,
    )
    if args.update and Path(args.out).exists():
        site = SiteConfig.load(args.out)
        site.calibration = cal
    else:
        site = SiteConfig(name=args.name, calibration=cal)
    if args.zone:
        site.zone = {"start_m": args.zone[0], "end_m": args.zone[1]}
    if args.lane:
        site.lanes = args.lane
    if args.speed_limit is not None:
        site.speed_limit_kmh = args.speed_limit
    rep = cal.report()
    print(json.dumps({"calibration": rep, "zone_in_view": site.zone_visibility()}, indent=2))
    site.save(args.out)
    print(f"wrote {args.out}")
    if args.check_image:
        import cv2

        if frame is None:
            sys.exit("--check-image needs --image or --video")
        cv2.imwrite(args.check_image, draw_overlay(frame, site))
        print(f"wrote {args.check_image} (yellow: clicked, magenta: re-projected survey points)")
    return 0


def cmd_run(args) -> int:
    from .calibration import SiteConfig
    from .pipeline import run_video, video_info

    site = SiteConfig.load(args.config)
    info = video_info(args.video)
    total = info["frames"] if args.max_frames is None else min(info["frames"], args.max_frames)
    step = max(1, total // 20)

    def progress(k):
        if not args.quiet and k % step == 0:
            print(f"  frame {k}/{total}", file=sys.stderr)

    out = run_video(
        args.video,
        site,
        args.out,
        review_sheets=args.review_sheets,
        max_frames=args.max_frames,
        progress=progress,
    )
    c = out.summary["counts"]
    print(
        f"{len(out.records)} records: {c['valid']} valid, {c['review']} for review, "
        f"{c['rejected']} rejected -> {args.out}"
    )
    if not site.zone_visibility()["all_inside"]:
        print("warning: the measurement zone is not fully inside the image", file=sys.stderr)
    return 0


def cmd_report(args) -> int:
    from .report import build_report

    rep = build_report(args.run_dir, args.out, n_boot=args.n_boot, title=args.title)
    g = rep["groups"].get("all", {})
    print(
        f"n={g.get('n')} mean={g.get('mean_kmh')} V85={g.get('v85_kmh')} "
        f"over-limit={g.get('share_over_limit')} flow={g.get('flow_veh_h')} veh/h -> "
        f"{Path(args.out or args.run_dir) / 'report.html'}"
    )
    return 0


def cmd_review(args) -> int:
    from .review import interactive_review, load_decisions, save_decisions

    if args.accept or args.reject:
        d = load_decisions(args.run_dir)
        for i in args.accept or []:
            d[int(i)] = "accept"
        for i in args.reject or []:
            d[int(i)] = "reject"
        save_decisions(args.run_dir, d, args.reviewer)
        print(f"{len(d)} decisions saved")
    else:
        d = interactive_review(args.run_dir, args.reviewer)
        print(f"{len(d)} decisions saved")
    return 0


def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def cmd_synth(args) -> int:
    from .synth.export import export_scenario
    from .synth.presets import PRESETS
    from .synth.scene import ScenarioConfig

    if args.scenario:
        cfg = ScenarioConfig.from_dict(json.loads(Path(args.scenario).read_text()))
    else:
        cfg = PRESETS[args.preset](seed=args.seed)
    if args.duration is not None:
        cfg.duration_s = args.duration
    if args.seed is not None:
        cfg.seed = args.seed
    upd = {}
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        upd[k] = _parse_value(v)
    if upd:
        cfg = cfg.replace(**upd)
    res = export_scenario(cfg, args.out, codec=args.codec)
    print(f"simulated {res['frames']} frames -> {res['video']}; site config {res['site']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="roadspeed", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="build a site config from image<->ground point pairs")
    src = c.add_mutually_exclusive_group()
    src.add_argument("--points", help="JSON list of {image: [u, v], ground: [X, Y]}")
    src.add_argument("--lanes", help="JSON lane-marking clicks (see docs/calibration.md)")
    c.add_argument(
        "--image", help="frame to click on (interactive) or to draw the check overlay on"
    )
    c.add_argument("--video", help="take the frame from this video")
    c.add_argument("--frame", type=int, default=0)
    c.add_argument("--survey", help="JSON list of {label, ground: [X, Y]}: click them in order")
    c.add_argument("--size", type=int, nargs=2, metavar=("W", "H"))
    c.add_argument("--out", required=True)
    c.add_argument(
        "--update", action="store_true", help="replace only the calibration of an existing --out"
    )
    c.add_argument("--name", default="site")
    c.add_argument("--zone", type=float, nargs=2, metavar=("START_M", "END_M"))
    c.add_argument(
        "--lane", type=_lane_arg, action="append", help="name:center_m:width_m:direction"
    )
    c.add_argument("--speed-limit", type=float)
    c.add_argument("--click-sigma", type=float, default=1.0, help="assumed click error, px")
    c.add_argument("--ground-sigma", type=float, default=0.03, help="assumed survey error, m")
    c.add_argument("--focal", type=float, help="known focal length in pixels (optional)")
    c.add_argument("--check-image", help="write an overlay of points, lanes and zone")
    c.set_defaults(func=cmd_calibrate)

    r = sub.add_parser("run", help="measure speeds in a video")
    r.add_argument("video")
    r.add_argument("--config", required=True)
    r.add_argument("--out", default="out")
    r.add_argument("--max-frames", type=int)
    r.add_argument(
        "--review-sheets",
        action="store_true",
        help="write anonymised review sheets for flagged records",
    )
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="statistics, figures and an HTML report for a run")
    rp.add_argument("run_dir")
    rp.add_argument("--out")
    rp.add_argument("--n-boot", type=int, default=2000)
    rp.add_argument("--title", default="Speed audit report")
    rp.set_defaults(func=cmd_report)

    rv = sub.add_parser("review", help="accept or reject flagged records")
    rv.add_argument("run_dir")
    rv.add_argument("--accept", nargs="*", type=int)
    rv.add_argument("--reject", nargs="*", type=int)
    rv.add_argument("--reviewer", default="")
    rv.set_defaults(func=cmd_review)

    s = sub.add_parser("synth", help="render a simulated scene with exact ground truth")
    s.add_argument("--out", required=True)
    s.add_argument("--preset", default="baseline", choices=["baseline", "four_lane", "small"])
    s.add_argument("--scenario", help="scenario JSON (overrides --preset)")
    s.add_argument("--seed", type=int)
    s.add_argument("--duration", type=float)
    s.add_argument("--set", action="append", help="dotted override, e.g. camera.height_m=8")
    s.add_argument("--codec", default="mp4v")
    s.set_defaults(func=cmd_synth)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
