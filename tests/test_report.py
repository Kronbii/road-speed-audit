import json

import numpy as np
import pytest

from roadspeed.report import build_report, group_summary, poisson_ci, speed_statistics


def test_poisson_ci_known_values():
    lo, hi = poisson_ci(10)
    assert lo == pytest.approx(4.795, abs=1e-3) and hi == pytest.approx(18.390, abs=1e-3)
    assert poisson_ci(0)[0] == 0.0


def test_speed_statistics():
    v = np.arange(1, 101, dtype=float)
    s = speed_statistics(v, 50)
    assert s["v85_kmh"] == pytest.approx(np.percentile(v, 85))
    assert s["share_over_limit"] == pytest.approx(0.5)


def test_group_ci_contains_point_and_widens_with_calibration():
    rng = np.random.default_rng(0)
    v = rng.normal(60, 10, 200)
    u = np.full(200, 0.5)
    tight = group_summary(v, u, np.ones(50), 50, 200, 1.0, n_boot=500)
    wide = group_summary(v, u, rng.normal(1, 0.03, 200), 50, 200, 1.0, n_boot=500)
    for g in (tight, wide):
        assert g["v85_kmh_ci"][0] <= g["v85_kmh"] <= g["v85_kmh_ci"][1]
    w = lambda g: g["mean_kmh_ci"][1] - g["mean_kmh_ci"][0]  # noqa: E731
    assert w(wide) > 1.5 * w(tight)
    assert tight["flow_veh_h"] == pytest.approx(200)


def test_build_report_from_files(tmp_path):
    from roadspeed.records import VehicleRecord, write_records_csv

    rng = np.random.default_rng(1)
    recs = []
    for i in range(40):
        v = float(rng.normal(55, 8))
        recs.append(
            VehicleRecord(
                record_id=i + 1,
                track_id=i + 1,
                segment=0,
                t_zone_in_s=i * 5.0,
                t_zone_out_s=i * 5.0 + 2,
                t_mid_s=i * 5.0 + 1,
                wall_time=None,
                lane="a" if i % 2 else "t",
                direction="away" if i % 2 else "towards",
                speed_kmh=v,
                u_kmh=0.6,
                u_cal_kmh=0.5,
                u_rand_kmh=0.3,
                ci95_low_kmh=v - 1.2,
                ci95_high_kmh=v + 1.2,
                n_points=40,
                coverage=0.95,
                residual_m=0.05,
                width_m=2.5,
                lateral_m=1.0,
                pixel_sigma_px=0.4,
                label="vehicle",
                status="review" if i == 3 else "valid",
                flags=["gap"] if i == 3 else [],
            )
        )
    write_records_csv(recs, tmp_path / "records.csv")
    summary = {
        "site": "test",
        "source": "x.mp4",
        "duration_s": 200.0,
        "speed_limit_kmh": 50,
        "zone": {"start_m": 20, "end_m": 50},
        "uncertainty": {"calibration_scale_samples": list(rng.normal(1, 0.01, 50))},
        "calibration": {
            "n_points": 9,
            "rms_px": 0.5,
            "rms_m": 0.05,
            "click_sigma_px_used": 1.0,
            "ground_sigma_m": 0.03,
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    rep = build_report(tmp_path, n_boot=300)
    assert rep["groups"]["all"]["n"] == 39  # the pending review record is excluded
    assert rep["groups"]["all"]["n_counted"] == 40
    assert (tmp_path / "report.html").read_text().startswith("<!doctype html>")
    assert (tmp_path / "figures" / "distribution.png").exists()
    # accepting the flagged record includes it
    from roadspeed.review import save_decisions

    save_decisions(tmp_path, {4: "accept"})
    rep2 = build_report(tmp_path, n_boot=300)
    assert rep2["groups"]["all"]["n"] == 40
