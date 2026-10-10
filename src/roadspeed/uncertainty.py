"""Monte-Carlo uncertainty for per-vehicle speeds.

Two sources are propagated separately and combined in quadrature:

* **Calibration** (shared by every vehicle of a run): the image clicks and the
  tape-measured ground coordinates of the calibration points are perturbed with
  Gaussian noise, the homography is re-fitted, and each track is re-projected and
  re-fitted. Because this error is common to all vehicles it does not average out
  in aggregate statistics; the per-draw ratio of perturbed to nominal speed is kept
  as a "calibration scale" sample so the report can carry it into the confidence
  intervals of mean speed and V85.
* **Pixel noise** of the contact point (detector jitter): Gaussian noise with a
  per-track standard deviation estimated from the fit residuals is added to the
  contact points, which are re-projected with the nominal homography and re-fitted.

The model is only as honest as its inputs: systematic effects that are not
noise (a reference point that drifts along the vehicle body, merged blobs) are
not captured. The synthetic experiments measure how well the intervals cover the
true error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import PinholeCamera, fit_homography, homography_jacobian
from .speed import RoadAxis, SpeedEstimate, fit_slope, project_to_road


@dataclass
class UncertaintyConfig:
    n_samples: int = 200
    click_sigma_px: float = 1.0  # std of a calibration click, per image axis
    ground_sigma_m: float = 0.03  # std of a surveyed ground coordinate, per axis
    min_pixel_sigma_px: float = 0.3  # floor on the per-track contact-point jitter
    seed: int = 12345
    use_reprojection_residuals: bool = True  # raise click sigma to what the residuals imply


def effective_click_sigma(cfg: UncertaintyConfig, img_pts, gnd_pts, H) -> float:
    """Click noise implied by the calibration residuals (needs >= 6 points), floored by cfg."""
    from .geometry import reprojection_report

    n = len(img_pts)
    sig = cfg.click_sigma_px
    if cfg.use_reprojection_residuals and n >= 6:
        rep = reprojection_report(H, img_pts, gnd_pts)
        # E[sum |r|^2] = sigma^2 (2n - 8) for an 8-parameter fit
        est = rep.rms_px * np.sqrt(n / (2 * n - 8))
        sig = max(sig, float(est))
    return sig


def calibration_draws(
    img_pts,
    gnd_pts,
    cfg: UncertaintyConfig,
    image_size: tuple[int, int],
    H_nominal: np.ndarray,
    need_camera: bool = False,
):
    """Perturbed homographies ``(M, 3, 3)`` (and camera centres ``(M, 3)`` if requested)."""
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 7]))
    img = np.asarray(img_pts, float)
    gnd = np.asarray(gnd_pts, float)
    sig_px = effective_click_sigma(cfg, img, gnd, H_nominal)
    Hs = np.empty((cfg.n_samples, 3, 3))
    Cs = np.full((cfg.n_samples, 3), np.nan) if need_camera else None
    for m in range(cfg.n_samples):
        ip = img + rng.normal(0, sig_px, img.shape)
        gp = gnd + rng.normal(0, cfg.ground_sigma_m, gnd.shape)
        Hs[m] = fit_homography(ip, gp, refine=len(img) > 4)
        if need_camera:
            try:
                cam = PinholeCamera.from_homography(
                    Hs[m], image_size[0], image_size[1], reference_ground_point=gp.mean(axis=0)
                )
                Cs[m] = cam.C
            except ValueError:
                pass
    return Hs, Cs, sig_px


def pixel_sigma(
    est: SpeedEstimate, H: np.ndarray, axis: RoadAxis, cfg: UncertaintyConfig, slope_a=None
):
    """Robust per-track contact-point jitter in pixels from along-road fit residuals."""
    idx = est.zone_idx
    uv = est.uv[idx]
    t = est.t[idx]
    s = axis.to_sl(project_to_road(H, uv))[:, 0]
    b, a, _ = fit_slope(t, s, "huber")
    r = s - (a + b * (t - t.mean()))
    J = homography_jacobian(H, uv)
    d = np.asarray(axis.direction)
    ds_dv = np.abs(J[:, 0, 1] * d[0] + J[:, 1, 1] * d[1])
    r_px = np.abs(r) / np.maximum(ds_dv, 1e-9)
    sig = 1.4826 * float(np.median(r_px)) if len(r_px) else 0.0
    return max(sig, cfg.min_pixel_sigma_px)


def propagate(
    estimates: list[SpeedEstimate],
    H: np.ndarray,
    axis: RoadAxis,
    fit: str,
    cfg: UncertaintyConfig,
    H_draws: np.ndarray,
    C_draws: np.ndarray | None = None,
    camera_center=None,
    contact_height_m: float = 0.0,
):
    """Fill per-vehicle uncertainties; return the calibration scale samples ``(M,)``.

    Returns ``(results, scale_samples)`` where ``results[i]`` is a dict with
    ``u_cal_kmh``, ``u_rand_kmh``, ``u_kmh`` and ``pixel_sigma_px`` for estimate ``i``
    (``None`` for estimates without a speed).
    """
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 11]))
    M = len(H_draws)
    results = []
    ratios = []
    for est in estimates:
        if est.speed_kmh is None or est.zone_idx is None or len(est.zone_idx) < 3:
            results.append(None)
            continue
        uv = est.uv[est.zone_idx]
        t = est.t[est.zone_idx]
        # calibration draws
        if contact_height_m and C_draws is not None:
            P = project_to_road(H_draws, uv, C_draws, contact_height_m)
        else:
            P = project_to_road(H_draws, uv)
        S = axis.to_sl(P)[..., 0]
        b, _, _ = fit_slope(t, S, fit)
        v_cal = np.abs(b) * 3.6
        ok = np.isfinite(v_cal)
        u_cal = float(np.std(v_cal[ok])) if ok.sum() > 2 else float("nan")
        ratios.append(np.where(ok, v_cal / est.speed_kmh, np.nan))
        # pixel noise draws
        sig = pixel_sigma(est, H, axis, cfg)
        noisy = uv[None, :, :] + rng.normal(0, sig, (M,) + uv.shape)
        Hb = np.broadcast_to(H, (M, 3, 3))
        if contact_height_m and camera_center is not None:
            Cb = np.broadcast_to(np.asarray(camera_center, float), (M, 3))
            P2 = project_to_road(Hb, noisy, Cb, contact_height_m)
        else:
            P2 = project_to_road(Hb, noisy)
        S2 = axis.to_sl(P2)[..., 0]
        b2, _, _ = fit_slope(t, S2, fit)
        u_rand = float(np.std(np.abs(b2) * 3.6))
        results.append(
            {
                "u_cal_kmh": u_cal,
                "u_rand_kmh": u_rand,
                "u_kmh": float(np.hypot(u_cal, u_rand)),
                "pixel_sigma_px": float(sig),
            }
        )
    if ratios:
        R = np.stack(ratios)
        scale = np.nanmedian(R, axis=0)
    else:
        scale = np.ones(M)
    return results, scale
