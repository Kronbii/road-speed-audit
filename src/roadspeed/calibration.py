"""Site configuration and calibration (JSON), lane-marking helper, interactive picker.

A *site* is one camera looking at one road segment. Its JSON file holds the
calibration correspondences (image pixels <-> ground metres), the along-road
axis, the measurement zone, lanes, the speed limit and the processing options.

Ground coordinates are metres in a right-handed frame on the road surface:
``X`` across the road (increasing to the right when looking along +Y), ``Y`` along
the road, increasing away from the camera.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .geometry import PinholeCamera, apply_homography, fit_homography, reprojection_report
from .speed import Lane, RoadAxis, SpeedConfig

SCHEMA_VERSION = 1


@dataclass
class Calibration:
    image_points: list
    ground_points: list
    image_size: list  # [width, height]
    click_sigma_px: float = 1.0
    ground_sigma_m: float = 0.03
    focal_px: float | None = None  # known focal length (optional; else recovered)
    note: str = ""

    def __post_init__(self):
        if len(self.image_points) != len(self.ground_points):
            raise ValueError("image_points and ground_points differ in length")
        if len(self.image_points) < 4:
            raise ValueError("calibration needs at least 4 point pairs")

    @property
    def img(self) -> np.ndarray:
        return np.asarray(self.image_points, float)

    @property
    def gnd(self) -> np.ndarray:
        return np.asarray(self.ground_points, float)

    def homography(self) -> np.ndarray:
        return fit_homography(self.img, self.gnd, refine=len(self.image_points) > 4)

    def report(self) -> dict:
        H = self.homography()
        rep = reprojection_report(H, self.img, self.gnd).as_dict()
        rep["n_points"] = len(self.image_points)
        if len(self.image_points) == 4:
            rep["warning"] = "4 points fit exactly: residuals are zero and say nothing; use >= 6"
        cam = self.camera()
        if cam is not None:
            rep["camera"] = cam.as_dict()
        return rep

    def camera(self) -> PinholeCamera | None:
        """Pinhole camera recovered from the homography (None if it cannot be)."""
        try:
            return PinholeCamera.from_homography(
                self.homography(),
                int(self.image_size[0]),
                int(self.image_size[1]),
                reference_ground_point=self.gnd.mean(axis=0),
                focal_px=self.focal_px,
            )
        except (ValueError, np.linalg.LinAlgError):
            return None


@dataclass
class SiteConfig:
    name: str
    calibration: Calibration
    zone: dict = field(default_factory=lambda: {"start_m": 20.0, "end_m": 50.0})
    road_axis: dict = field(default_factory=lambda: {"origin": [0.0, 0.0], "direction": [0.0, 1.0]})
    lanes: list = field(default_factory=list)  # [{"name", "center_m", "width_m", "direction"}]
    speed_limit_kmh: float | None = None
    detector: dict = field(default_factory=lambda: {"backend": "bgsub"})
    tracker: dict = field(default_factory=dict)
    speed: dict = field(default_factory=dict)  # overrides for SpeedConfig
    uncertainty: dict = field(default_factory=dict)
    roi: str | list = "auto"  # "auto", "none", or an image polygon [[u, v], ...]
    timestamps: str = "fps"  # "fps" (frame index / fps) or "container" (decoder timestamps)
    fps: float | None = None  # override the container frame rate
    start_time: str | None = None  # ISO time of the first frame, if known
    prime_frames: int = 25  # frames for the median background initialisation
    schema_version: int = SCHEMA_VERSION

    # -------------------------------------------------------------- derived
    def axis(self) -> RoadAxis:
        return RoadAxis(tuple(self.road_axis["origin"]), tuple(self.road_axis["direction"]))

    def lane_objs(self) -> list[Lane]:
        return [Lane(**ln) for ln in self.lanes]

    def speed_config(self) -> SpeedConfig:
        d = dict(self.speed)
        d.setdefault("zone_start_m", float(self.zone["start_m"]))
        d.setdefault("zone_end_m", float(self.zone["end_m"]))
        return SpeedConfig(**d)

    def roi_mask(
        self, height_m: float = 4.5, margin_before_m: float = 25.0, margin_after_m: float = 60.0
    ):
        """Binary image mask of where vehicles can appear (None = whole frame)."""
        import cv2

        W, Hh = (int(v) for v in self.calibration.image_size)
        if self.roi == "none":
            return None
        if isinstance(self.roi, list):
            mask = np.zeros((Hh, W), np.uint8)
            cv2.fillPoly(mask, [np.round(np.asarray(self.roi)).astype(np.int32)], 1)
            return mask
        cam = self.calibration.camera()
        if cam is None or not self.lanes:
            return None
        ax = self.axis()
        lanes = self.lane_objs()
        l0 = min(ln.center_m - ln.width_m / 2 for ln in lanes) - 1.5
        l1 = max(ln.center_m + ln.width_m / 2 for ln in lanes) + 1.5
        z0 = min(self.zone["start_m"], self.zone["end_m"]) - margin_before_m
        z1 = max(self.zone["start_m"], self.zone["end_m"]) + margin_after_m
        d = np.asarray(ax.direction)
        n = np.array([d[1], -d[0]])
        o = np.asarray(ax.origin)
        pts3 = [
            [*(o + s * d + lat * n), z]
            for s in np.linspace(z0, z1, 61)
            for lat in (l0, l1)
            for z in (0.0, height_m)
        ]
        P = np.asarray(pts3)
        Xc = cam.to_camera(P)
        P = P[Xc[:, 2] > 1.0]  # keep the part of the region in front of the camera
        if len(P) < 3:
            return None
        uv, _ = cam.project(P)
        uv = np.clip(uv, -4 * W, 5 * W)
        hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
        mask = np.zeros((Hh, W), np.uint8)
        cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
        return mask

    def zone_visibility(self) -> dict:
        """Where the zone corners fall in the image (to warn about an out-of-view zone)."""
        H = self.calibration.homography()
        G = np.linalg.inv(H)
        ax = self.axis()
        d = np.asarray(ax.direction)
        n = np.array([d[1], -d[0]])
        o = np.asarray(ax.origin)
        lanes = self.lane_objs() or [Lane("all", 0.0, 7.0)]
        l0 = min(ln.center_m - ln.width_m / 2 for ln in lanes)
        l1 = max(ln.center_m + ln.width_m / 2 for ln in lanes)
        out = {}
        W, Hh = self.calibration.image_size
        all_in = True
        # the sign of the homogeneous coordinate of points in front of the camera
        ref = G @ np.array([*self.calibration.gnd.mean(axis=0), 1.0])
        front = np.sign(ref[2])
        for name, s in (("start", self.zone["start_m"]), ("end", self.zone["end_m"])):
            for side, lat in (("left", l0), ("right", l1)):
                g = o + s * d + lat * n
                q = G @ np.array([g[0], g[1], 1.0])
                uv = q[:2] / q[2]
                inside = bool(q[2] * front > 0 and 0 <= uv[0] < W and 0 <= uv[1] < Hh)
                all_in &= inside
                out[f"{name}_{side}"] = [round(float(uv[0]), 1), round(float(uv[1]), 1), inside]
        out["all_inside"] = all_in
        return out

    # ------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, d: dict) -> SiteConfig:
        d = dict(d)
        d["calibration"] = Calibration(**d["calibration"])
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown site-config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path) -> SiteConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Lane-marking helper
# ---------------------------------------------------------------------------


def points_from_lane_markings(
    lines: list[dict], dash_length_m: float, dash_gap_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Ground coordinates for clicks on dashed lane lines of known geometry.

    ``lines`` is a list of ``{"x_m": lateral position of the line, "phase_m": 0.0,
    "points": [{"uv": [u, v], "dash": k, "end": "near"|"far"}, ...]}``, where dash
    ``k = 0`` is the dash nearest to the camera that you clicked, and ``near`` /
    ``far`` is the end of that dash. Lane width and dash geometry must be measured on
    site with a tape; do not assume a national standard.

    With clicks on parallel lines only, the along-road offset between lines
    (``phase_m``) cannot be recovered from the image: any value gives an equally good
    fit, differing by a shear of the ground plane. That shear does not change
    distances measured *along* a lane, so along-lane speeds are unaffected; it does
    affect lateral offsets of lane changes. Measure ``phase_m`` with the tape (or add
    one point on a transverse marking) if it matters.
    """
    period = dash_length_m + dash_gap_m
    img, gnd = [], []
    for ln in lines:
        x = float(ln["x_m"])
        ph = float(ln.get("phase_m", 0.0))
        for p in ln["points"]:
            y = (
                ph
                + int(p["dash"]) * period
                + (0.0 if p.get("end", "near") == "near" else dash_length_m)
            )
            img.append([float(p["uv"][0]), float(p["uv"][1])])
            gnd.append([x, y])
    return np.asarray(img), np.asarray(gnd)


# ---------------------------------------------------------------------------
# Overlay for visual checking
# ---------------------------------------------------------------------------


def draw_overlay(image: np.ndarray, site: SiteConfig) -> np.ndarray:
    """Draw calibration points (clicked vs re-projected), zone and lanes on a copy."""
    import cv2

    out = image.copy()
    cal = site.calibration
    H = cal.homography()
    G = np.linalg.inv(H)
    reproj = apply_homography(G, cal.gnd)
    for (u, v), (ru, rv) in zip(cal.img, reproj, strict=True):
        cv2.circle(out, (int(round(u)), int(round(v))), 6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(
            out, (int(round(ru)), int(round(rv))), (255, 0, 255), cv2.MARKER_CROSS, 12, 2
        )
    ax = site.axis()
    d = np.asarray(ax.direction)
    n = np.array([d[1], -d[0]])
    o = np.asarray(ax.origin)
    lanes = site.lane_objs()
    s_vals = np.linspace(
        min(site.zone["start_m"], site.zone["end_m"]) - 10,
        max(site.zone["start_m"], site.zone["end_m"]) + 30,
        60,
    )
    for ln in lanes:
        for lat in (ln.center_m - ln.width_m / 2, ln.center_m + ln.width_m / 2):
            g = o[None, :] + s_vals[:, None] * d[None, :] + lat * n[None, :]
            uv = apply_homography(G, g)
            cv2.polylines(
                out, [np.round(uv).astype(np.int32)], False, (255, 200, 0), 1, cv2.LINE_AA
            )
    l0 = min((ln.center_m - ln.width_m / 2 for ln in lanes), default=-3.5)
    l1 = max((ln.center_m + ln.width_m / 2 for ln in lanes), default=3.5)
    for s in (site.zone["start_m"], site.zone["end_m"]):
        g = np.array([o + s * d + l0 * n, o + s * d + l1 * n])
        uv = apply_homography(G, g)
        cv2.line(
            out,
            tuple(np.round(uv[0]).astype(int)),
            tuple(np.round(uv[1]).astype(int)),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Interactive point picker (OpenCV HighGUI)
# ---------------------------------------------------------------------------


def pick_points(
    image: np.ndarray, labels: list[str] | None = None, window: str = "roadspeed calibrate"
):
    """Click calibration points on ``image``.

    Left click adds a point, ``u`` undoes the last one, ``Enter`` finishes, ``Esc``
    aborts. If ``labels`` is given, the window asks for each labelled surveyed point
    in turn and finishes automatically after the last one. A magnified inset around
    the cursor helps place clicks to a pixel. Returns an ``(N, 2)`` array.
    """
    import cv2

    pts: list[tuple[float, float]] = []
    cursor = [0, 0]

    def on_mouse(event, x, y, flags, param):
        cursor[0], cursor[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            vis = image.copy()
            for i, (u, v) in enumerate(pts):
                cv2.circle(vis, (int(u), int(v)), 5, (0, 255, 255), 1, cv2.LINE_AA)
                name = labels[i] if labels and i < len(labels) else str(i + 1)
                cv2.putText(
                    vis,
                    name,
                    (int(u) + 6, int(v) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            if labels:
                if len(pts) >= len(labels):
                    break
                msg = f"click point {labels[len(pts)]}  (u: undo, Esc: abort)"
            else:
                msg = f"{len(pts)} points - click to add, u: undo, Enter: done (>=4), Esc: abort"
            cv2.putText(
                vis, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                vis, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
            # magnifier
            x, y = cursor
            r = 20
            h, w = image.shape[:2]
            x0, y0 = min(max(x - r, 0), w - 2 * r), min(max(y - r, 0), h - 2 * r)
            patch = image[y0 : y0 + 2 * r, x0 : x0 + 2 * r]
            if patch.size:
                zoom = cv2.resize(patch, (8 * r, 8 * r), interpolation=cv2.INTER_NEAREST)
                cv2.drawMarker(
                    zoom, (4 * (x - x0), 4 * (y - y0)), (0, 0, 255), cv2.MARKER_CROSS, 16, 1
                )
                vis[h - 8 * r :, w - 8 * r :] = zoom
            cv2.imshow(window, vis)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                raise KeyboardInterrupt("calibration aborted")
            if k == ord("u") and pts:
                pts.pop()
            if k in (10, 13) and len(pts) >= 4 and not labels:
                break
    finally:
        cv2.destroyWindow(window)
    return np.asarray(pts)
