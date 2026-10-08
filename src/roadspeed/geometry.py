"""Plane geometry: homographies and a pinhole camera model.

Conventions used throughout the package
----------------------------------------
* Ground (world) frame, metres: ``X`` across the road, ``Y`` along the road, ``Z`` up.
  The road surface is the plane ``Z = 0``.
* Camera frame (OpenCV): ``x`` right, ``y`` down, ``z`` forward (optical axis).
* Image coordinates are pixels with the origin at the top-left corner of the
  top-left pixel's centre convention used by OpenCV (pixel ``(0, 0)`` centre at 0.0).
* ``H`` (unless stated otherwise) maps homogeneous *image* points to *ground* points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Homography estimation
# ---------------------------------------------------------------------------


def _as_points(pts) -> np.ndarray:
    a = np.asarray(pts, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) array of points, got shape {a.shape}")
    return a


def normalization_transform(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: translate centroid to 0, scale mean distance to sqrt(2)."""
    pts = _as_points(pts)
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / d if d > 0 else 1.0
    return np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])


def apply_homography(H: np.ndarray, pts) -> np.ndarray:
    """Apply a homography to points.

    ``H`` may be ``(3, 3)`` with ``pts`` ``(N, 2)``, or batched ``(M, 3, 3)`` with
    ``pts`` ``(N, 2)`` or ``(M, N, 2)``. Returns ``(N, 2)`` or ``(M, N, 2)``.
    """
    H = np.asarray(H, dtype=float)
    p = np.asarray(pts, dtype=float)
    if H.ndim == 2:
        ph = np.concatenate([p, np.ones(p.shape[:-1] + (1,))], axis=-1)
        q = ph @ H.T
    else:
        if p.ndim == 2:
            p = np.broadcast_to(p, (H.shape[0],) + p.shape)
        ph = np.concatenate([p, np.ones(p.shape[:-1] + (1,))], axis=-1)
        q = np.einsum("mij,mnj->mni", H, ph)
    return q[..., :2] / q[..., 2:3]


def fit_homography_dlt(src, dst) -> np.ndarray:
    """Normalised direct linear transform, ``dst ~ H @ src``. Needs >= 4 points."""
    src = _as_points(src)
    dst = _as_points(dst)
    if len(src) != len(dst):
        raise ValueError("src and dst must have the same number of points")
    if len(src) < 4:
        raise ValueError("a homography needs at least 4 point pairs")
    Ts = normalization_transform(src)
    Td = normalization_transform(dst)
    s = apply_homography(Ts, src)
    d = apply_homography(Td, dst)
    n = len(s)
    A = np.zeros((2 * n, 9))
    x, y = s[:, 0], s[:, 1]
    u, v = d[:, 0], d[:, 1]
    A[0::2, 0:3] = np.stack([-x, -y, -np.ones(n)], axis=1)
    A[0::2, 6:9] = np.stack([u * x, u * y, u], axis=1)
    A[1::2, 3:6] = np.stack([-x, -y, -np.ones(n)], axis=1)
    A[1::2, 6:9] = np.stack([v * x, v * y, v], axis=1)
    _, sv, vt = np.linalg.svd(A)
    if n == 4 and sv[-2] < 1e-10:
        raise ValueError("degenerate point configuration (three or more points collinear?)")
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def _refine_image_error(H_i2g: np.ndarray, img_pts: np.ndarray, gnd_pts: np.ndarray) -> np.ndarray:
    """Refine an image->ground homography by minimising image-space reprojection error.

    Image clicks are the noisy quantity in a site calibration, so the residual is
    measured in pixels: ``|| H^-1 ground - image ||``.
    """
    from scipy.optimize import least_squares

    G0 = np.linalg.inv(H_i2g)
    G0 = G0 / G0[2, 2]

    def resid(p):
        G = np.append(p, 1.0).reshape(3, 3)
        return (apply_homography(G, gnd_pts) - img_pts).ravel()

    sol = least_squares(resid, G0.ravel()[:8], method="lm", xtol=1e-12, ftol=1e-12)
    G = np.append(sol.x, 1.0).reshape(3, 3)
    H = np.linalg.inv(G)
    return H / H[2, 2]


def fit_homography(img_pts, gnd_pts, refine: bool = True) -> np.ndarray:
    """Image->ground homography from >= 4 correspondences (pixels -> metres)."""
    img = _as_points(img_pts)
    gnd = _as_points(gnd_pts)
    H = fit_homography_dlt(img, gnd)
    if refine and len(img) > 4:
        H = _refine_image_error(H, img, gnd)
    return H


@dataclass
class ReprojectionReport:
    rms_px: float
    max_px: float
    rms_m: float
    max_m: float
    per_point_px: list[float]
    per_point_m: list[float]

    def as_dict(self) -> dict:
        r = lambda v: float(round(v, 4))  # noqa: E731
        return {
            "rms_px": r(self.rms_px),
            "max_px": r(self.max_px),
            "rms_m": r(self.rms_m),
            "max_m": r(self.max_m),
            "per_point_px": [r(v) for v in self.per_point_px],
            "per_point_m": [r(v) for v in self.per_point_m],
        }


def reprojection_report(H_i2g, img_pts, gnd_pts) -> ReprojectionReport:
    """Residuals of a calibration, in pixels (ground->image) and metres (image->ground).

    With exactly 4 points the fit is exact and residuals are zero; use >= 6 points
    spread over the measurement zone so the residuals mean something.
    """
    img = _as_points(img_pts)
    gnd = _as_points(gnd_pts)
    e_m = np.linalg.norm(apply_homography(H_i2g, img) - gnd, axis=1)
    e_px = np.linalg.norm(apply_homography(np.linalg.inv(H_i2g), gnd) - img, axis=1)
    return ReprojectionReport(
        rms_px=float(np.sqrt(np.mean(e_px**2))),
        max_px=float(e_px.max()),
        rms_m=float(np.sqrt(np.mean(e_m**2))),
        max_m=float(e_m.max()),
        per_point_px=e_px.tolist(),
        per_point_m=e_m.tolist(),
    )


def homography_jacobian(H_i2g: np.ndarray, uv) -> np.ndarray:
    """Jacobian d(ground)/d(image) at image points, shape ``(N, 2, 2)`` (metres per pixel)."""
    uv = _as_points(uv)
    ph = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
    q = ph @ H_i2g.T
    w = q[:, 2]
    J = np.empty((len(uv), 2, 2))
    for i in range(2):
        for j in range(2):
            J[:, i, j] = (H_i2g[i, j] * w - q[:, i] * H_i2g[2, j]) / w**2
    return J


# ---------------------------------------------------------------------------
# Pinhole camera
# ---------------------------------------------------------------------------


def rotation_from_angles(yaw_deg: float, tilt_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """World->camera rotation for a camera looking along +Y.

    ``yaw`` turns the optical axis towards +X, ``tilt`` pitches it down towards the
    road, ``roll`` rotates the image clockwise about the optical axis.
    """
    psi, th, ph = np.radians([yaw_deg, tilt_deg, roll_deg])
    f = np.array([np.sin(psi) * np.cos(th), np.cos(psi) * np.cos(th), -np.sin(th)])
    r = np.array([np.cos(psi), -np.sin(psi), 0.0])
    d = np.cross(f, r)
    r2 = np.cos(ph) * r + np.sin(ph) * d
    d2 = -np.sin(ph) * r + np.cos(ph) * d
    return np.stack([r2, d2, f])


@dataclass
class PinholeCamera:
    """Ideal pinhole camera (no lens distortion), world->camera ``x_c = R (X - C)``."""

    K: np.ndarray
    R: np.ndarray
    C: np.ndarray
    width: int
    height: int

    @classmethod
    def from_pose(
        cls,
        focal_px: float,
        width: int,
        height: int,
        position,
        yaw_deg: float = 0.0,
        tilt_deg: float = 0.0,
        roll_deg: float = 0.0,
        cx: float | None = None,
        cy: float | None = None,
    ) -> PinholeCamera:
        cx = (width - 1) / 2.0 if cx is None else cx
        cy = (height - 1) / 2.0 if cy is None else cy
        K = np.array([[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]])
        R = rotation_from_angles(yaw_deg, tilt_deg, roll_deg)
        return cls(K=K, R=R, C=np.asarray(position, dtype=float), width=width, height=height)

    @staticmethod
    def focal_from_hfov(width: int, hfov_deg: float) -> float:
        return (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)

    @property
    def t(self) -> np.ndarray:
        return -self.R @ self.C

    @property
    def focal_px(self) -> float:
        return float(self.K[0, 0])

    @property
    def camera_height(self) -> float:
        return float(self.C[2])

    @property
    def forward(self) -> np.ndarray:
        return self.R[2]

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arcsin(-self.forward[2])))

    @property
    def yaw_deg(self) -> float:
        f = self.forward
        return float(np.degrees(np.arctan2(f[0], f[1])))

    def to_camera(self, Xw) -> np.ndarray:
        Xw = np.asarray(Xw, dtype=float)
        return (Xw - self.C) @ self.R.T

    def project(self, Xw) -> tuple[np.ndarray, np.ndarray]:
        """Project world points ``(N, 3)`` -> pixels ``(N, 2)`` and depths ``(N,)``."""
        Xc = self.to_camera(Xw)
        z = Xc[..., 2]
        uvw = Xc @ self.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uvw[..., :2] / uvw[..., 2:3]
        return uv, z

    def rays(self, uv) -> np.ndarray:
        """Unit ray directions in world coordinates for pixels ``(N, 2)``."""
        uv = _as_points(uv)
        ph = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
        d = ph @ np.linalg.inv(self.K).T @ self.R
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def backproject_to_plane(self, uv, z: float = 0.0) -> np.ndarray:
        """Intersect pixel rays with the horizontal plane ``Z = z``. Returns ``(N, 2)`` X, Y.

        Rays that do not hit the plane in front of the camera return NaN.
        """
        d = self.rays(uv)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - self.C[2]) / d[:, 2]
        s = np.where(s > 0, s, np.nan)
        P = self.C[None, :] + s[:, None] * d
        return P[:, :2]

    def ground_to_image_homography(self) -> np.ndarray:
        G = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        return G / G[2, 2]

    def image_to_ground_homography(self) -> np.ndarray:
        H = np.linalg.inv(self.ground_to_image_homography())
        return H / H[2, 2]

    def horizon_row(self, u: float | None = None) -> float:
        """Image row of the horizon at column ``u`` (default: centre)."""
        u = self.K[0, 2] if u is None else u
        # vanishing line of plane Z=0 is l = K^-T R e3 ... solve l . (u, v, 1) = 0
        n_c = self.R @ np.array([0.0, 0.0, 1.0])
        line = np.linalg.inv(self.K).T @ n_c
        return float(-(line[0] * u + line[2]) / line[1])

    @classmethod
    def from_homography(
        cls,
        H_i2g: np.ndarray,
        width: int,
        height: int,
        cx: float | None = None,
        cy: float | None = None,
        reference_ground_point=(0.0, 0.0),
        focal_px: float | None = None,
    ) -> PinholeCamera:
        """Recover focal length and pose from an image->ground homography.

        Assumes square pixels, zero skew and a known principal point (default: image
        centre). The focal length comes from the orthonormality of the first two
        rotation columns; pass ``focal_px`` if it is known from the lens instead.
        """
        cx = (width - 1) / 2.0 if cx is None else cx
        cy = (height - 1) / 2.0 if cy is None else cy
        G = np.linalg.inv(H_i2g)
        T = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
        Gp = np.linalg.inv(T) @ G
        g1, g2 = Gp[:, 0], Gp[:, 1]
        if focal_px is None:
            # Two constraints on w = 1/f^2:  w*a1 + b1 = 0 and w*a2 + b2 = 0
            a = np.array(
                [g1[0] * g2[0] + g1[1] * g2[1], (g1[0] ** 2 + g1[1] ** 2) - (g2[0] ** 2 + g2[1] ** 2)]
            )
            b = np.array([g1[2] * g2[2], g1[2] ** 2 - g2[2] ** 2])
            # Ordinary least squares in w. Both residuals have the same units
            # (lambda^2), and an equation that is degenerate for the pose (e.g. the
            # orthogonality constraint for a camera with zero yaw) has a ~ b ~ 0 and
            # simply drops out.
            w = -float(a @ b) / float(a @ a)
            if not np.isfinite(w) or w <= 0:
                raise ValueError("cannot recover a focal length from this homography")
            focal_px = 1.0 / np.sqrt(w)
        K = np.array([[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]])
        M = np.linalg.inv(K) @ G
        lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
        ref = np.array([reference_ground_point[0], reference_ground_point[1], 1.0])
        if (M @ ref)[2] * lam < 0:
            lam = -lam
        r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
        r3 = np.cross(r1, r2)
        Rm = np.column_stack([r1, r2, r3])
        U, _, Vt = np.linalg.svd(Rm)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
        C = -R.T @ t
        if C[2] < 0:
            raise ValueError(
                "recovered camera lies below the ground plane: the ground axes are probably "
                "left-handed (use X across the road, Y along it, Z up)"
            )
        return cls(K=K, R=R, C=C, width=width, height=height)

    def as_dict(self) -> dict:
        return {
            "focal_px": round(self.focal_px, 3),
            "cx": round(float(self.K[0, 2]), 3),
            "cy": round(float(self.K[1, 2]), 3),
            "position_m": [round(float(v), 4) for v in self.C],
            "height_m": round(self.camera_height, 4),
            "tilt_deg": round(self.tilt_deg, 4),
            "yaw_deg": round(self.yaw_deg, 4),
            "width": int(self.width),
            "height": int(self.height),
        }
