"""Portfolio plates for road-speed-audit, drawn from the committed results.

A square and a 4:3 crop of the simulated hero frame (with its "SIMULATED SCENE" label redrawn
so a crop can never drop it), and 4:3 plates for the distance and contact-point experiments.
Run from the repository root:

    python experiments/portfolio_plates.py <outdir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from common import setup_matplotlib  # noqa: E402

BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"


def label(im: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(im)
    size = max(18, im.width // 36)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    x, y = size * 0.7, size * 0.6
    box = draw.textbbox((x, y), "SIMULATED SCENE", font=font)
    pad = size * 0.35
    draw.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=(20, 22, 24))
    draw.text((x, y), "SIMULATED SCENE", font=font, fill=(255, 255, 255))
    return im


def crops(out: Path) -> None:
    # The source frame already carries the label top-left; crop below and right of it, then redraw.
    im = Image.open(ROOT / "figures" / "hero_frame.png").convert("RGB")
    w, h = im.size
    sq = im.crop(((w - h) // 2, 0, (w + h) // 2, h))
    label(sq).save(out / "plate.png", optimize=True)
    w43 = h * 4 // 3
    wide = im.crop(((w - w43) // 2, 0, (w + w43) // 2, h))
    label(wide).save(out / "scene.png", optimize=True)


def distance(out: Path) -> None:
    rows = json.loads((ROOT / "results" / "e02_distance.json").read_text())["conditions"]
    rows = [r for r in rows if r["error_valid_kmh"].get("p95_abs") is not None]
    x = [r["level"]["zone_centre_m"] for r in rows]
    p95 = [r["error_valid_kmh"]["p95_abs"] for r in rows]
    lo = [p - r["error_valid_kmh"]["p95_abs_ci"][0] for p, r in zip(p95, rows, strict=True)]
    hi = [r["error_valid_kmh"]["p95_abs_ci"][1] - p for p, r in zip(p95, rows, strict=True)]
    valid = [100 * r["outcomes"]["valid"] / r["gt_vehicles"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.errorbar(x, p95, yerr=[lo, hi], color=BLUE, marker="o", ms=7, lw=2.2, capsize=4)
    for i, (xi, yi, v) in enumerate(zip(x, p95, valid, strict=True)):
        # labels left of the steep far points, below-right of the near ones, clear of the bars
        far = i >= len(x) - 2
        ax.annotate(
            f"{v:.0f} % measured",
            (xi, yi),
            xytext=(-12, 8) if far else (8, -16),
            textcoords="offset points",
            ha="right" if far else "left",
            fontsize=10,
            color=INK2,
        )
    ax.set_ylim(-1.2, None)
    ax.axhline(2.0, color=ORANGE, ls="--", lw=1.4)
    ax.text(x[0], 2.25, "2 km/h", color=ORANGE, fontsize=10)
    ax.set_xlabel("distance of the measurement zone from the pole (m)", fontsize=12)
    ax.set_ylabel("95th-percentile speed error (km/h)", fontsize=12)
    ax.set_title("Error grows with distance (simulated video)", loc="left", fontsize=13)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(out / "distance.png", dpi=200)
    plt.close(fig)


def contact(out: Path) -> None:
    rows = json.loads((ROOT / "results" / "e12_contact.json").read_text())["rows"]
    names = {
        "bottom edge, sub-pixel refined (default)": "bottom edge,\nsub-pixel (default)",
        "bottom edge, raw blob boundary": "bottom edge,\nraw blob",
        "box centre, height-corrected (0.75 m)": "box centre,\nheight-corrected",
        "box centre, ground homography": "box centre,\nno correction",
    }
    rows = sorted(rows, key=lambda r: r["error_valid_kmh"]["p95_abs"], reverse=True)
    labels = [names.get(r["model"], r["model"]) for r in rows]
    vals = [r["error_valid_kmh"]["p95_abs"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colors = [BLUE if "default" in r["model"] else "#9aa3ad" for r in rows]
    ax.barh(labels, vals, color=colors, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f} km/h", va="center", fontsize=11, color=INK2)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("95th-percentile speed error (km/h)", fontsize=12)
    ax.set_title("Which point on the vehicle to track (simulated)", loc="left", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "contact.png", dpi=200)
    plt.close(fig)


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else ROOT / "figures" / "portfolio"
    out.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()
    crops(out)
    distance(out)
    contact(out)
    print("wrote plates to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
