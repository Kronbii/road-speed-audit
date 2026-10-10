"""Vehicle shapes for the synthetic renderer.

Vehicles are unions of convex hexahedra ("parts"): a lower body with ground
clearance, a cabin or cargo box, and wheels. Dimensions are typical values chosen
for simulation only (they are assumptions, not a survey of any national fleet).
The local frame has ``x`` to the right of the vehicle, ``y`` forward, ``z`` up,
with the origin at the centre of the footprint on the road.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# (mean length, mean width, mean height, sd length) in metres
CLASS_DIMS = {
    "car": (4.4, 1.80, 1.47, 0.30),
    "suv": (4.8, 1.90, 1.75, 0.25),
    "van": (5.2, 2.00, 2.20, 0.30),
    "truck": (9.0, 2.50, 3.40, 1.20),
    "bus": (12.0, 2.55, 3.20, 0.50),
    "motorcycle": (2.1, 0.80, 1.55, 0.10),
}

DEFAULT_MIX = {
    "car": 0.64,
    "suv": 0.14,
    "van": 0.08,
    "truck": 0.06,
    "bus": 0.03,
    "motorcycle": 0.05,
}

# BGR body colours and rough relative frequencies (assumed, for variety only)
PALETTE = [
    ((232, 232, 230), 0.27),  # white
    ((186, 188, 190), 0.15),  # silver
    ((118, 118, 120), 0.13),  # grey
    ((34, 34, 38), 0.15),  # black
    ((140, 72, 38), 0.08),  # blue
    ((40, 42, 168), 0.08),  # red
    ((62, 98, 54), 0.04),  # green
    ((150, 176, 198), 0.05),  # beige
    ((40, 150, 205), 0.05),  # orange/yellow
]

GLASS = (48, 40, 36)
TYRE = (28, 28, 30)

# Face order for a hexahedron with vertices 0-3 bottom, 4-7 top, both in order
# (x0,y0) (x1,y0) (x1,y1) (x0,y1).
HEX_FACES = np.array(
    [
        [0, 3, 2, 1],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # rear  (y0)
        [2, 3, 7, 6],  # front (y1)
        [3, 0, 4, 7],  # left  (x0)
        [1, 2, 6, 5],  # right (x1)
    ]
)


def hexahedron(bottom, top, z0, z1) -> np.ndarray:
    """8 vertices from bottom rect ``(x0, x1, y0, y1)`` at ``z0`` and top rect at ``z1``."""
    bx0, bx1, by0, by1 = bottom
    tx0, tx1, ty0, ty1 = top
    return np.array(
        [
            [bx0, by0, z0],
            [bx1, by0, z0],
            [bx1, by1, z0],
            [bx0, by1, z0],
            [tx0, ty0, z1],
            [tx1, ty0, z1],
            [tx1, ty1, z1],
            [tx0, ty1, z1],
        ],
        dtype=float,
    )


@dataclass
class Part:
    verts: np.ndarray  # (8, 3) local coordinates
    face_colors: list  # 6 BGR tuples (unshaded)
    layer: int = 1  # draw order within a vehicle: 0 wheels, 1 body, 2 upper parts


@dataclass
class VehicleModel:
    cls: str
    length: float
    width: float
    height: float
    clearance: float
    color: tuple
    parts: list[Part] = field(default_factory=list)

    def all_vertices(self) -> np.ndarray:
        return np.concatenate([p.verts for p in self.parts], axis=0)


def _box(x0, x1, y0, y1, z0, z1):
    return hexahedron((x0, x1, y0, y1), (x0, x1, y0, y1), z0, z1)


def _wheels(L, W, r, axles_y, inset=0.10, tyre_w=0.22):
    parts = []
    for y in axles_y:
        for side in (-1, 1):
            xo = side * (W / 2 - inset)
            xi = side * (W / 2 - inset - tyre_w)
            x0, x1 = min(xo, xi), max(xo, xi)
            parts.append(Part(_box(x0, x1, y - r, y + r, 0.0, 2 * r), [TYRE] * 6, layer=0))
    return parts


def build_vehicle(cls: str, rng: np.random.Generator) -> VehicleModel:
    """Sample a vehicle of class ``cls`` with randomised dimensions and colour."""
    Lm, Wm, Hm, Ls = CLASS_DIMS[cls]
    L = float(np.clip(rng.normal(Lm, Ls), 0.8 * Lm, 1.25 * Lm))
    W = float(np.clip(rng.normal(Wm, 0.05), 0.92 * Wm, 1.08 * Wm))
    H = float(np.clip(rng.normal(Hm, 0.06 * Hm), 0.9 * Hm, 1.12 * Hm))
    cols = [c for c, _ in PALETTE]
    probs = np.array([p for _, p in PALETTE])
    color = tuple(int(v) for v in cols[rng.choice(len(cols), p=probs / probs.sum())])
    if cls in ("bus", "truck") and rng.random() < 0.5:
        color = (232, 232, 230)
    hx, hy = W / 2, L / 2
    parts: list[Part] = []

    if cls in ("car", "suv"):
        clr = 0.16 if cls == "car" else 0.21
        belt = clr + (0.62 if cls == "car" else 0.60) * (H - clr)
        wb_f = 0.85 if cls == "car" else 0.90
        parts.append(Part(_box(-hx, hx, -hy, hy, clr, belt), [color] * 6))
        hood = 1.25 if cls == "car" else 1.05
        trunk = 0.55 if cls == "car" else 0.20
        bot = (-0.47 * W, 0.47 * W, -hy + trunk, hy - hood)
        top = (-0.40 * W, 0.40 * W, -hy + trunk + 0.35, hy - hood - 0.60)
        cab_cols = [GLASS, color, GLASS, GLASS, GLASS, GLASS]
        parts.append(Part(hexahedron(bot, top, belt, H), cab_cols, layer=2))
        r = 0.31 if cls == "car" else 0.35
        parts += _wheels(L, W, r, (-hy + wb_f, hy - wb_f))
    elif cls == "van":
        clr = 0.20
        belt = 1.05
        parts.append(Part(_box(-hx, hx, -hy, hy, clr, belt), [color] * 6))
        bot = (-0.49 * W, 0.49 * W, -hy, hy - 0.75)
        top = (-0.47 * W, 0.47 * W, -hy, hy - 1.35)
        cab_cols = [color, color, color, GLASS, color, color]
        parts.append(Part(hexahedron(bot, top, belt, H), cab_cols, layer=2))
        parts += _wheels(L, W, 0.34, (-hy + 0.95, hy - 0.95))
    elif cls == "truck":
        clr = 0.45
        cab_len = 2.3
        cab_color = tuple(int(v) for v in cols[rng.choice(len(cols), p=probs / probs.sum())])
        cab_faces = [cab_color] * 3 + [GLASS] + [cab_color] * 2
        parts.append(Part(_box(-hx, hx, hy - cab_len, hy, clr, 2.95), cab_faces, layer=2))
        chassis = _box(-hx * 0.9, hx * 0.9, -hy + 0.4, hy - cab_len, clr, clr + 0.55)
        parts.append(Part(chassis, [TYRE] * 6, layer=1))
        cargo = _box(-hx, hx, -hy, hy - cab_len - 0.25, clr + 0.6, H)
        parts.append(Part(cargo, [color] * 6, layer=2))
        parts += _wheels(L, W, 0.50, (hy - 1.4, -hy + 2.6, -hy + 1.4), inset=0.05, tyre_w=0.35)
    elif cls == "bus":
        clr = 0.35
        cols_bus = [color, color, color, GLASS, color, color]
        parts.append(Part(_box(-hx, hx, -hy, hy, clr, H), cols_bus))
        parts += _wheels(L, W, 0.50, (hy - 2.6, -hy + 3.0), inset=0.05, tyre_w=0.30)
    elif cls == "motorcycle":
        clr = 0.28
        parts += [
            Part(_box(-0.07, 0.07, 0.35, 1.05, 0.0, 0.64), [TYRE] * 6, layer=0),
            Part(_box(-0.07, 0.07, -1.05, -0.35, 0.0, 0.64), [TYRE] * 6, layer=0),
            Part(_box(-0.20, 0.20, -0.75, 0.75, clr, 0.92), [color] * 6, layer=1),
            Part(_box(-0.40, 0.40, 0.45, 0.60, 0.92, 1.05), [TYRE] * 6, layer=2),  # handlebars
            Part(_box(-0.24, 0.24, -0.45, 0.15, 0.90, H), [(60, 55, 50)] * 6, layer=2),  # rider
        ]
    else:  # pragma: no cover - guarded by CLASS_DIMS
        raise ValueError(f"unknown vehicle class {cls!r}")

    return VehicleModel(
        cls=cls, length=L, width=W, height=H, clearance=clr, color=color, parts=parts
    )
