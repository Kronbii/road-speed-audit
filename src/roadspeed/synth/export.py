"""Write a simulated scenario to disk: video, ground truth and a site configuration."""

from __future__ import annotations

import csv
from pathlib import Path

from ..records import dump_json
from .render import Renderer
from .scene import ScenarioConfig


def export_scenario(
    cfg: ScenarioConfig,
    out_dir,
    video_name: str = "video.mp4",
    codec: str = "mp4v",
    click_sigma_px: float = 0.7,
    ground_sigma_m: float = 0.02,
    progress=None,
) -> dict:
    """Render ``cfg`` to ``out_dir``.

    Files: ``video.mp4`` (simulated frames), ``ground_truth.json`` (every vehicle's
    true speed, lane, class and zone-crossing times), ``gt_boxes.csv`` (per-frame
    visible boxes), ``scenario.json`` and ``site.json`` (a simulated field
    calibration of this camera, ready for ``roadspeed run``).
    """
    import cv2

    from ..evaluate import site_from_scenario

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = Renderer(cfg)
    W, H = cfg.camera.width, cfg.camera.height
    writer = cv2.VideoWriter(
        str(out / video_name), cv2.VideoWriter_fourcc(*codec), cfg.camera.fps, (W, H)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer with codec {codec!r}")
    with open(out / "gt_boxes.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame", "t_s", "vid", "x1", "y1", "x2", "y2", "visibility", "truncated"])
        for k in range(r.n_frames):
            frame, anns = r.render(k)
            writer.write(frame)
            for a in anns:
                if a.bbox_visible is None:
                    continue
                w.writerow(
                    [
                        k,
                        f"{r.frame_time(k):.6f}",
                        a.vid,
                        *a.bbox_visible,
                        f"{a.visibility:.4f}",
                        int(a.truncated),
                    ]
                )
            if progress is not None:
                progress(k)
    writer.release()
    dump_json(
        {
            "note": "simulated scene; every number here is ground truth of the simulation",
            "camera": r.cam.as_dict(),
            "zone": {"start_m": cfg.zone.start_m, "end_m": cfg.zone.end_m},
            "vehicles": r.ground_truth_vehicles(),
        },
        out / "ground_truth.json",
    )
    (out / "scenario.json").write_text(cfg.to_json() + "\n")
    site = site_from_scenario(cfg, click_sigma_px=click_sigma_px, ground_sigma_m=ground_sigma_m)
    site.save(out / "site.json")
    return {"video": str(out / video_name), "frames": r.n_frames, "site": str(out / "site.json")}
