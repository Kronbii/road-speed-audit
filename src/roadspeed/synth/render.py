"""Synthetic road-scene renderer (numpy + OpenCV, no GPU, no downloads).

The road is a procedurally textured plane rendered once per scenario with
analytic anti-aliasing of the lane markings; vehicles are drawn per frame as
shaded convex parts with a painter's algorithm, cast and contact shadows, motion
blur (sub-frame averaging), lighting drift, sensor noise and JPEG compression.
An ID buffer rendered alongside gives per-frame ground-truth boxes and
visibility. Frames are random-access and deterministic for a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .scene import ScenarioConfig, SimVehicle, build_camera, generate_traffic
from .vehicles import HEX_FACES, Part, VehicleModel, _box

NEAR = 0.3  # near clipping plane, metres
SIDEWALK_M = 2.5
SHIFT = 4  # sub-pixel bits for cv2 polygon drawing
SCALE = float(1 << SHIFT)


@dataclass
class FrameAnnotation:
    vid: int
    bbox_full: tuple  # (x1, y1, x2, y2) of the unoccluded projection, clipped to the image
    bbox_visible: tuple | None  # bounding box of visible pixels (None if fully occluded)
    visibility: float  # visible pixels / unoccluded pixels inside the image
    truncated: bool  # the projection extends beyond the image border


def _face_normals(verts: np.ndarray) -> np.ndarray:
    """Outward unit normals for the 6 faces of a convex hexahedron."""
    c = verts.mean(axis=0)
    n = np.empty((6, 3))
    for i, f in enumerate(HEX_FACES):
        p = verts[f]
        v = np.cross(p[2] - p[0], p[3] - p[1])
        if np.dot(v, p.mean(axis=0) - c) < 0:
            v = -v
        n[i] = v / (np.linalg.norm(v) + 1e-12)
    return n


def _clip_near(Pc: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a camera-frame polygon against z >= NEAR."""
    out = []
    n = len(Pc)
    for i in range(n):
        a, b = Pc[i], Pc[(i + 1) % n]
        ina, inb = a[2] >= NEAR, b[2] >= NEAR
        if ina:
            out.append(a)
        if ina != inb:
            s = (NEAR - a[2]) / (b[2] - a[2])
            out.append(a + s * (b - a))
    return np.array(out) if out else np.zeros((0, 3))


class _Object:
    """A drawable object (vehicle or static occluder) with precomputed geometry."""

    def __init__(self, parts: list[Part]):
        order = sorted(range(len(parts)), key=lambda i: parts[i].layer)
        self.parts = [parts[i] for i in order]
        self.local_verts = np.stack([p.verts for p in self.parts])  # (P, 8, 3)
        self.local_normals = np.stack([_face_normals(p.verts) for p in self.parts])  # (P, 6, 3)
        self.colors = np.array([p.face_colors for p in self.parts], dtype=float)  # (P, 6, 3)
        self.layers = [p.layer for p in self.parts]


class Renderer:
    """Render frames and ground truth for a :class:`ScenarioConfig`."""

    def __init__(self, cfg: ScenarioConfig, vehicles: list[SimVehicle] | None = None):
        self.cfg = cfg
        self.cam = build_camera(cfg)
        self.W, self.Hh = cfg.camera.width, cfg.camera.height
        self.vehicles = generate_traffic(cfg) if vehicles is None else vehicles
        self.fps = float(cfg.camera.fps)
        self.n_frames = int(round(cfg.duration_s * self.fps))
        el, az = np.radians([cfg.lighting.sun_elevation_deg, cfg.lighting.sun_azimuth_deg])
        self.sun = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
        self._K = self.cam.K
        self._R = self.cam.R
        self._C = self.cam.C
        self._vobjs = {v.vid: _Object(v.model.parts) for v in self.vehicles}
        self._occluders = [self._make_occluder(o) for o in cfg.occluders]
        self.background = self._render_background()
        self._background_u8 = np.clip(np.rint(self.background), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ setup
    def _make_occluder(self, o: dict):
        x, y = float(o["x"]), float(o["y"])
        w, d, h = float(o.get("w", 0.3)), float(o.get("d", 0.3)), float(o.get("h", 8.0))
        col = tuple(o.get("color", (105, 108, 110)))
        part = Part(_box(-w / 2, w / 2, -d / 2, d / 2, 0.0, h), [col] * 6, layer=1)
        return (np.array([x, y]), _Object([part]))

    def _render_background(self) -> np.ndarray:
        cam = self.cam
        W, Hh = self.W, self.Hh
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(Hh, dtype=float))
        uv = np.stack([u.ravel(), v.ravel()], axis=1)
        rays = cam.rays(uv)
        C = cam.C
        hit = rays[:, 2] < -1e-9
        s = np.where(hit, -C[2] / np.where(hit, rays[:, 2], -1.0), np.inf)
        X = C[0] + s * rays[:, 0]
        Y = C[1] + s * rays[:, 1]
        ground = hit & (s < 4000.0)
        # pixel footprint on the ground from the image->ground Jacobian
        Hm = cam.image_to_ground_homography()
        q = uv @ Hm[:, :2].T + Hm[:, 2]
        w = q[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            dXdu = (Hm[0, 0] * w - q[:, 0] * Hm[2, 0]) / w**2
            dXdv = (Hm[0, 1] * w - q[:, 0] * Hm[2, 1]) / w**2
            dYdu = (Hm[1, 0] * w - q[:, 1] * Hm[2, 0]) / w**2
            dYdv = (Hm[1, 1] * w - q[:, 1] * Hm[2, 1]) / w**2
        fx = np.abs(dXdu) + np.abs(dXdv)
        fy = np.abs(dYdu) + np.abs(dYdv)
        fx = np.where(ground, np.clip(fx, 1e-4, 1e4), 1.0)
        fy = np.where(ground, np.clip(fy, 1e-4, 1e4), 1.0)
        X = np.where(ground, X, 0.0)
        Y = np.where(ground, Y, 0.0)
        col = self._ground_texture(X, Y, fx, fy)
        # haze with distance
        haze = np.array([196.0, 190.0, 182.0])
        a = 1.0 - np.exp(-np.where(ground, s, 1e9) / 1500.0)
        col = col * (1 - a[:, None]) + haze * a[:, None]
        # sky gradient by ray elevation
        elev = np.arcsin(np.clip(rays[:, 2], -1, 1))
        tt = np.clip(elev / 0.45, 0, 1)[:, None]
        sky = np.array([232.0, 214.0, 192.0]) * (1 - tt) + np.array([205.0, 150.0, 95.0]) * tt
        col = np.where(ground[:, None], col, sky)
        img = col.reshape(Hh, W, 3)
        img = self._draw_hills(img, ground.reshape(Hh, W), rays[:, 2].reshape(Hh, W))
        img = self._draw_buildings(img)
        return img.astype(np.float32)

    # --------------------------------------------------------------- scenery
    def _draw_hills(self, img, ground, ray_z):
        """Distant hills above the horizon (sky pixels only)."""
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, 6]))
        W = self.W
        # angular height profile (radians) as smoothed noise over viewing azimuth
        az = np.arctan2(
            *self.cam.rays(np.stack([np.arange(W, dtype=float), np.full(W, self.Hh / 2)], 1))[
                :, :2
            ].T
        )
        grid = np.linspace(-np.pi, np.pi, 721)
        prof = np.cumsum(rng.normal(0, 1, grid.size))
        prof = np.convolve(prof - prof.mean(), np.ones(25) / 25, mode="same")
        prof = 0.012 + 0.022 * (prof - prof.min()) / (np.ptp(prof) + 1e-9)
        h = np.interp(az, grid, prof)  # (W,)
        elev = np.arcsin(np.clip(ray_z, -1, 1))
        mask = (~ground) & (elev >= 0) & (elev <= h[None, :])
        hill = np.array([168.0, 162.0, 150.0])
        tt = np.clip(elev / np.maximum(h[None, :], 1e-6), 0, 1)[..., None]
        shade = hill * (0.92 + 0.08 * tt)
        return np.where(mask[..., None], shade, img)

    def _draw_buildings(self, img):
        """Rows of simple buildings set back from both roadsides, baked into the background.

        They sit outside the region between the camera and the road, so vehicles are
        always in front of them and no per-frame depth ordering is needed.
        """
        road = self.cfg.road
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, 4]))
        hw = road.width_m / 2 + road.shoulder_m + SIDEWALK_M
        x_right = max(hw, self._C[0]) + 2.0
        facades = [
            (205, 214, 222),
            (182, 196, 212),
            (170, 178, 186),
            (196, 205, 200),
            (150, 170, 196),
        ]
        blds = []
        for side, x0 in ((-1, -hw - 1.5), (1, x_right)):
            y = -20.0 + rng.uniform(0, 8)
            while y < 420.0:
                w = rng.uniform(9, 22)
                if rng.random() < 0.8:
                    d = rng.uniform(8, 14)
                    h = rng.choice(
                        [3.2, 6.4, 9.6, 12.8, 16.0, 19.2], p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.1]
                    )
                    setback = rng.uniform(0, 4)
                    xa = x0 + side * setback
                    xb = xa + side * d
                    col = facades[rng.integers(len(facades))]
                    blds.append((min(xa, xb), max(xa, xb), y, y + w, float(h), col, side))
                y += w + rng.uniform(1.0, 8.0)
        canvas = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        haze = np.array([196.0, 190.0, 182.0])
        order = sorted(
            blds,
            key=lambda b: -np.hypot((b[0] + b[1]) / 2 - self._C[0], (b[2] + b[3]) / 2 - self._C[1]),
        )
        for x0, x1, y0, y1, h, col, side in order:
            verts = _box(x0, x1, y0, y1, 0.0, h)
            obj = _Object([Part(verts, [col] * 6, layer=1)])
            Wv, Wn = obj.local_verts[0], obj.local_normals[0]
            polys, sil, Pc = self._part_polys(Wv, Wn)
            if not polys:
                continue
            dist = float(np.linalg.norm(Pc.mean(axis=0)))
            a = 1.0 - np.exp(-dist / 1500.0)
            cols = self._shade(obj.colors[0], Wn, 1.0) * (1 - a) + haze * a
            for i, uv in polys:
                pts = np.round(uv * SCALE)
                if np.abs(pts).max() > 2**30:
                    continue
                cv2.fillConvexPoly(
                    canvas,
                    pts.astype(np.int32),
                    tuple(float(c) for c in cols[i]),
                    cv2.LINE_AA,
                    SHIFT,
                )
            # windows on the facade facing the road
            face = 5 if side < 0 else 4  # +x face for left-side buildings, -x for right side
            xf = x1 if side < 0 else x0
            if not any(i == face for i, _ in polys) or dist > 220:
                continue
            wc = np.clip(cols[face] * 0.45, 0, 255)
            wc = tuple(float(c) for c in wc * (1 - a) + haze * a)
            for zf in np.arange(1.0, h - 1.0, 3.2):
                for yw in np.arange(y0 + 1.2, y1 - 1.6, 3.0):
                    P = np.array(
                        [
                            [xf, yw, zf],
                            [xf, yw + 1.4, zf],
                            [xf, yw + 1.4, zf + 1.5],
                            [xf, yw, zf + 1.5],
                        ]
                    )
                    Pc2 = self._to_cam(P)
                    if np.any(Pc2[:, 2] < NEAR):
                        continue
                    pts = np.round(self._proj(Pc2) * SCALE)
                    if np.abs(pts).max() > 2**30:
                        continue
                    cv2.fillConvexPoly(canvas, pts.astype(np.int32), wc, cv2.LINE_AA, SHIFT)
        return canvas.astype(np.float64)

    def _noise_tiles(self, rng: np.random.Generator, n: int) -> list[np.ndarray]:
        tiles = []
        for _ in range(n):
            t = rng.standard_normal((256, 256)).astype(np.float32)
            t = cv2.GaussianBlur(np.pad(t, 4, mode="wrap"), (0, 0), 0.8)[4:-4, 4:-4]
            tiles.append(t / t.std())
        return tiles

    def _ground_texture(self, X, Y, fx, fy) -> np.ndarray:
        road = self.cfg.road
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, 3]))
        hw = road.width_m / 2
        sh = road.shoulder_m
        fp = np.maximum(fx, fy)

        def box_cov(lo, hi):
            """Fraction of the pixel footprint (in X) inside [lo, hi]."""
            a = np.clip(X + fx / 2, lo, hi) - np.clip(X - fx / 2, lo, hi)
            return np.clip(a / fx, 0, 1)

        # noise octaves (wavelength m, amplitude DN)
        octaves = [(0.04, 7.0), (0.15, 5.0), (0.6, 4.0), (2.5, 3.0), (10.0, 3.0), (40.0, 2.0)]
        tiles = self._noise_tiles(rng, len(octaves))
        noise = np.zeros_like(X)
        for (lam, amp), tile in zip(octaves, tiles, strict=True):
            att = np.clip(1.5 - fp / lam, 0.0, 1.0)
            if not att.any():
                continue
            off = rng.uniform(0, 256, size=2)
            mx = np.mod(X / lam + off[0], 256.0).astype(np.float32)
            my = np.mod(Y / lam + off[1], 256.0).astype(np.float32)
            smp = cv2.remap(
                tile,
                mx.reshape(self.Hh, self.W),
                my.reshape(self.Hh, self.W),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            ).ravel()
            noise += amp * att * smp

        asphalt = np.array([92.0, 92.0, 95.0])
        gravel = np.array([120.0, 126.0, 130.0])
        walk = np.array([168.0, 172.0, 174.0])  # pavement beyond the shoulder
        verge = np.array([96.0, 128.0, 140.0])  # dusty ground (BGR)
        paint = np.array([222.0, 224.0, 225.0])

        c_road = box_cov(-hw, hw)
        c_sh = box_cov(-hw - sh, hw + sh) - c_road
        c_walk = box_cov(-hw - sh - SIDEWALK_M, hw + sh + SIDEWALK_M) - c_road - c_sh
        c_verge = 1.0 - c_road - c_sh - c_walk
        col = (
            c_road[:, None] * asphalt
            + c_sh[:, None] * gravel
            + c_walk[:, None] * walk
            + c_verge[:, None] * verge
            + (noise * (0.7 + 0.6 * c_verge))[:, None]
        )
        # darker wheel tracks in each lane
        tracks = np.zeros_like(X)
        for xc in road.lane_centers():
            for o in (-0.85, 0.85):
                tracks += np.exp(-0.5 * ((X - xc - o) / 0.3) ** 2)
        col -= (5.0 * np.clip(tracks, 0, 1) * c_road)[:, None]

        # lane markings
        lw = road.line_width_m
        P = road.dash_length_m + road.dash_gap_m
        Ld = road.dash_length_m

        def dash_cov():
            def F(y):
                return np.floor(y / P) * Ld + np.minimum(np.mod(y, P), Ld)

            return np.clip((F(Y + fy / 2) - F(Y - fy / 2)) / fy, 0, 1)

        dashes = dash_cov()
        cov = np.zeros_like(X)
        for xe in (-hw + lw / 2 + 0.1, hw - lw / 2 - 0.1):  # solid edge lines
            cov = np.maximum(cov, box_cov(xe - lw / 2, xe + lw / 2))
        n = road.n_lanes
        for k in range(1, n):
            xl = -hw + k * road.lane_width_m
            centre = road.two_way and k == n // 2
            if centre and road.lanes_per_direction > 1:  # double solid centre line
                for o in (-0.12, 0.12):
                    cov = np.maximum(cov, box_cov(xl + o - lw / 2, xl + o + lw / 2))
            else:
                cov = np.maximum(cov, box_cov(xl - lw / 2, xl + lw / 2) * dashes)
        cov *= 0.92
        col = col * (1 - cov[:, None]) + paint * cov[:, None]
        return np.clip(col, 0, 255)

    # --------------------------------------------------------------- per frame
    def frame_time(self, k: int) -> float:
        return k / self.fps

    def gain(self, t: float) -> float:
        L = self.cfg.lighting
        g = L.gain * (1.0 + L.drift_amplitude * np.sin(2 * np.pi * t / L.drift_period_s))
        for tj, fac in L.jumps:
            if t >= tj:
                g *= fac
        return float(g)

    def _to_cam(self, Pw: np.ndarray) -> np.ndarray:
        return (Pw - self._C) @ self._R.T

    def _proj(self, Pc: np.ndarray) -> np.ndarray:
        q = Pc @ self._K.T
        return q[:, :2] / q[:, 2:3]

    def _objects_at(self, t: float):
        """Objects in view at time t: list of (dist, kind, id, world_verts, world_normals, obj)."""
        road = self.cfg.road
        out = []
        for v in self.vehicles:
            y = float(v.y_at(t))
            if not (road.visible_near_m - 10 <= y <= road.visible_far_m):
                continue
            o = self._vobjs[v.vid]
            d = v.direction
            Lv = o.local_verts
            Wv = np.empty_like(Lv)
            Wv[..., 0] = v.x_m + d * Lv[..., 0]
            Wv[..., 1] = y + d * Lv[..., 1]
            Wv[..., 2] = Lv[..., 2]
            Wn = o.local_normals.copy()
            Wn[..., 0] *= d
            Wn[..., 1] *= d
            dist = float(np.hypot(v.x_m - self._C[0], y - self._C[1]))
            out.append((dist, "vehicle", v.vid, Wv, Wn, o))
        for pos, o in self._occluders:
            Wv = o.local_verts.copy()
            Wv[..., 0] += pos[0]
            Wv[..., 1] += pos[1]
            dist = float(np.hypot(pos[0] - self._C[0], pos[1] - self._C[1]))
            out.append((dist, "occluder", -1, Wv, o.local_normals, o))
        out.sort(key=lambda r: -r[0])
        return out

    def _part_polys(self, Wv_part: np.ndarray, Wn_part: np.ndarray):
        """Visible face polygons (pixels) of one part, plus a silhouette if unclipped."""
        Pc = self._to_cam(Wv_part)
        centres = Wv_part[HEX_FACES].mean(axis=1)
        facing = np.einsum("ij,ij->i", Wn_part, self._C[None, :] - centres) > 1e-9
        if not facing.any():
            return [], None, Pc
        if np.all(Pc[:, 2] >= NEAR):
            uv = self._proj(Pc)
            polys = [(i, uv[HEX_FACES[i]]) for i in np.nonzero(facing)[0]]
            sil = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
            return polys, sil, Pc
        if np.all(Pc[:, 2] < NEAR):
            return [], None, Pc
        polys = []
        for i in np.nonzero(facing)[0]:
            cp = _clip_near(Pc[HEX_FACES[i]])
            if len(cp) >= 3:
                polys.append((i, self._proj(cp)))
        return polys, None, Pc

    def _shade(self, colors: np.ndarray, normals: np.ndarray, g: float) -> np.ndarray:
        lam = np.clip(normals @ self.sun, 0, None)
        sky = 0.5 + 0.5 * normals[:, 2]
        k = 0.42 + 0.18 * sky + 0.52 * lam
        return np.clip(colors * k[:, None] * g, 0, 255)

    def _ground_poly_to_image(self, gp: np.ndarray):
        Pw = np.column_stack([gp, np.zeros(len(gp))])
        Pc = self._to_cam(Pw)
        if np.all(Pc[:, 2] >= NEAR):
            return self._proj(Pc)
        cp = _clip_near(Pc)
        return self._proj(cp) if len(cp) >= 3 else None

    def _draw_shadows(self, canvas: np.ndarray, objs) -> None:
        """Darken the road under and beside vehicles (cast + contact shadows), in place."""
        L = self.cfg.lighting
        cast, contact = [], []
        for _, kind, _, Wv, _, _ in objs:
            if kind != "vehicle":
                continue
            P = Wv.reshape(-1, 3)
            if L.shadows and self.sun[2] > 0.05:
                g = P[:, :2] - (P[:, 2:3] / self.sun[2]) * self.sun[None, :2]
                hull = cv2.convexHull(g.astype(np.float32)).reshape(-1, 2).astype(float)
                cast.append(hull)
            fp = P[:, :2]
            lo, hi = fp.min(axis=0), fp.max(axis=0)
            contact.append(
                np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
            )
        polys = [(c, int(255 * L.shadow_strength)) for c in cast]
        polys += [(c, int(255 * 0.62)) for c in contact]
        mask = None
        for gp, val in polys:
            ip = self._ground_poly_to_image(gp)
            if ip is None:
                continue
            pts = np.round(ip * SCALE)
            if np.abs(pts).max() > 2**30:
                continue
            if mask is None:
                mask = np.zeros((self.Hh, self.W), np.uint8)
            cv2.fillConvexPoly(mask, pts.astype(np.int32), val, cv2.LINE_AA, SHIFT)
        if mask is None:
            return
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            return
        x0, y0 = max(0, x - 3), max(0, y - 3)
        x1, y1 = min(self.W, x + w + 3), min(self.Hh, y + h + 3)
        m = cv2.GaussianBlur(mask[y0:y1, x0:x1], (5, 5), 0)
        keep = cv2.merge([255 - m] * 3)
        roi = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = cv2.multiply(roi, keep, scale=1.0 / 255.0)

    def _draw_objects(self, canvas: np.ndarray, objs, g: float) -> None:
        for _, _, _, Wv, Wn, o in objs:
            pending = []
            for p in range(len(o.parts)):
                polys, sil, Pc = self._part_polys(Wv[p], Wn[p])
                if not polys:
                    continue
                d = float(np.linalg.norm(Pc.mean(axis=0)))
                pending.append((o.layers[p], -d, p, polys, sil))
            pending.sort(key=lambda r: (r[0], r[1]))
            for _, _, p, polys, sil in pending:
                cols = self._shade(o.colors[p], Wn[p], g)
                if sil is not None:
                    mean = cols[[i for i, _ in polys]].mean(axis=0)
                    pts = np.round(sil * SCALE).astype(np.int32)
                    cv2.fillConvexPoly(
                        canvas, pts, tuple(float(c) for c in mean), cv2.LINE_AA, SHIFT
                    )
                for i, uv in polys:
                    pts = np.round(uv * SCALE)
                    if np.abs(pts).max() > 2**30:
                        continue
                    cv2.fillConvexPoly(
                        canvas,
                        pts.astype(np.int32),
                        tuple(float(c) for c in cols[i]),
                        cv2.LINE_AA,
                        SHIFT,
                    )

    def _render_color(self, t: float) -> np.ndarray:
        g = self.gain(t)
        objs = self._objects_at(t)
        if g == 1.0:
            canvas = self._background_u8.copy()
        else:
            canvas = cv2.convertScaleAbs(self.background, alpha=g)
        if objs:
            self._draw_shadows(canvas, objs)
            self._draw_objects(canvas, objs, g)
        return canvas

    def render(self, k: int, annotate: bool = True):
        """Render frame ``k``. Returns ``(frame_bgr_uint8, annotations)``."""
        cam = self.cfg.camera
        t = self.frame_time(k)
        nsub = max(1, int(cam.blur_subframes)) if cam.exposure_s > 0 else 1
        if nsub == 1:
            frame = self._render_color(t)
        else:
            acc = np.zeros((self.Hh, self.W, 3), np.float32)
            for j in range(nsub):
                tj = t + cam.exposure_s * ((j + 0.5) / nsub - 0.5)
                cv2.accumulate(self._render_color(tj), acc)
            frame = cv2.convertScaleAbs(acc, alpha=1.0 / nsub)
        if cam.noise_sigma > 0:
            frame = cv2.add(frame, self._noise_for_frame(k), dtype=cv2.CV_8U)
        if cam.jpeg_quality is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(cam.jpeg_quality)])
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        anns = self.annotate(t) if annotate else []
        return frame, anns

    _NOISE_BANK = 8
    _NOISE_PAD = 64

    def _noise_for_frame(self, k: int) -> np.ndarray:
        """Gaussian sensor noise for frame ``k`` (int16), deterministic per frame.

        Drawing 2.8 M normal samples per 720p frame dominates the render time, so a
        small bank of padded noise fields is generated once and each frame takes a
        randomly offset window of a randomly chosen field. Per-pixel noise is still
        Gaussian with the requested sigma and practically uncorrelated in time.
        """
        if not hasattr(self, "_noise_bank"):
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, 2]))
            p = self._NOISE_PAD
            shape = (self._NOISE_BANK, self.Hh + p, self.W + p, 3)
            sig = float(self.cfg.camera.noise_sigma)
            self._noise_bank = np.rint(rng.standard_normal(shape, dtype=np.float32) * sig).astype(
                np.int16
            )
        r = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, 2, k]))
        j = int(r.integers(self._NOISE_BANK))
        dy, dx = (int(v) for v in r.integers(0, self._NOISE_PAD, size=2))
        return self._noise_bank[j, dy : dy + self.Hh, dx : dx + self.W]

    # ------------------------------------------------------------ ground truth
    def vehicle_mask(self, t: float) -> np.ndarray:
        """Boolean image mask of visible vehicle pixels at time ``t`` (for figures)."""
        return self._idbuf(t)[0] > 0

    def annotate(self, t: float) -> list[FrameAnnotation]:
        """Ground-truth boxes and visibility at time ``t`` from an ID buffer."""
        idbuf, faces_by_slot, slot_vid = self._idbuf(t)
        return self._annotations(idbuf, faces_by_slot, slot_vid)

    def _idbuf(self, t: float):
        objs = self._objects_at(t)
        idbuf = np.zeros((self.Hh, self.W), np.uint16)
        faces_by_slot: dict[int, list] = {}
        slot_vid: dict[int, int] = {}
        slot = 0
        for _, kind, vid, Wv, Wn, o in objs:
            all_polys = []
            for p in range(len(o.parts)):
                polys, _, _ = self._part_polys(Wv[p], Wn[p])
                all_polys += [uv for _, uv in polys]
            if kind == "vehicle":
                slot += 1
                val = slot
                slot_vid[slot] = vid
                faces_by_slot[slot] = all_polys
            else:
                val = 0
            for uv in all_polys:
                pts = np.round(uv * SCALE)
                if np.abs(pts).max() > 2**30:
                    continue
                cv2.fillConvexPoly(idbuf, pts.astype(np.int32), val, cv2.LINE_8, SHIFT)
        return idbuf, faces_by_slot, slot_vid

    def _annotations(self, idbuf, faces_by_slot, slot_vid) -> list[FrameAnnotation]:
        anns = []
        for s, polys in faces_by_slot.items():
            if not polys:
                continue
            allp = np.concatenate(polys)
            x1, y1 = allp.min(axis=0)
            x2, y2 = allp.max(axis=0)
            truncated = bool(x1 < 0 or y1 < 0 or x2 > self.W - 1 or y2 > self.Hh - 1)
            cx1, cy1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
            cx2, cy2 = int(min(self.W - 1, np.ceil(x2))), int(min(self.Hh - 1, np.ceil(y2)))
            if cx2 < cx1 or cy2 < cy1:
                continue
            crop = idbuf[cy1 : cy2 + 1, cx1 : cx2 + 1]
            ys, xs = np.nonzero(crop == s)
            full = np.zeros(crop.shape, np.uint8)
            off = np.array([cx1, cy1], float)
            for uv in polys:
                pts = np.round((uv - off) * SCALE)
                if np.abs(pts).max() > 2**30:
                    continue
                cv2.fillConvexPoly(full, pts.astype(np.int32), 1, cv2.LINE_8, SHIFT)
            n_full = int(full.sum())
            if n_full == 0:
                continue
            if len(xs):
                bv = (
                    cx1 + int(xs.min()),
                    cy1 + int(ys.min()),
                    cx1 + int(xs.max()),
                    cy1 + int(ys.max()),
                )
            else:
                bv = None
            anns.append(
                FrameAnnotation(
                    vid=slot_vid[s],
                    bbox_full=(
                        float(max(x1, 0)),
                        float(max(y1, 0)),
                        float(min(x2, self.W - 1)),
                        float(min(y2, self.Hh - 1)),
                    ),
                    bbox_visible=bv,
                    visibility=len(xs) / n_full,
                    truncated=truncated,
                )
            )
        anns.sort(key=lambda a: a.vid)
        return anns

    def frames(self, start: int = 0, stop: int | None = None, annotate: bool = True):
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        for k in range(start, stop):
            frame, anns = self.render(k, annotate=annotate)
            yield k, self.frame_time(k), frame, anns

    # ----------------------------------------------------------- vehicle facts
    def vehicle_by_id(self, vid: int) -> SimVehicle:
        return self.vehicles[vid]

    def ground_truth_vehicles(self) -> list[dict]:
        """Per-vehicle ground truth for vehicles whose centre crosses the whole zone."""
        z = self.cfg.zone
        t_end = self.cfg.duration_s
        out = []
        for v in self.vehicles:
            t_in = v.time_at_y(z.start_m if v.direction > 0 else z.end_m)
            t_out = v.time_at_y(z.end_m if v.direction > 0 else z.start_m)
            m: VehicleModel = v.model
            out.append(
                {
                    "vid": v.vid,
                    "class": m.cls,
                    "lane": v.lane,
                    "direction": "away" if v.direction > 0 else "towards",
                    "speed_kmh": round(v.speed_mps * 3.6, 6),
                    "x_m": round(v.x_m, 4),
                    "t_zone_in": round(t_in, 6),
                    "t_zone_out": round(t_out, 6),
                    "length_m": round(m.length, 3),
                    "width_m": round(m.width, 3),
                    "height_m": round(m.height, 3),
                    "clearance_m": round(m.clearance, 3),
                    "in_window": bool(t_in >= self.cfg.warmup_s and t_out <= t_end),
                }
            )
        return out
