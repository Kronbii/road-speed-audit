"""Traffic-engineering summary of a run: mean speed, V85, share over the limit, flow.

Statistics use valid records plus flagged records a reviewer accepted.
Confidence intervals (95 %) come from a bootstrap that resamples vehicles,
perturbs each speed by its random (pixel-noise) uncertainty, and multiplies by a
calibration scale drawn from the run's Monte-Carlo calibration samples, so the
shared calibration error is carried into every interval rather than averaged
away. Flow intervals are exact Poisson (Garwood) intervals on the count.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

from .records import dump_json, read_records_csv
from .review import load_decisions


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def speed_statistics(v: np.ndarray, limit: float | None) -> dict:
    v = np.asarray(v, float)
    out = {
        "mean_kmh": float(np.mean(v)) if len(v) else float("nan"),
        "sd_kmh": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
        "v50_kmh": _pct(v, 50),
        "v85_kmh": _pct(v, 85),
    }
    if limit is not None:
        out["share_over_limit"] = float(np.mean(v > limit)) if len(v) else float("nan")
        out["share_over_limit_plus10"] = float(np.mean(v > limit + 10)) if len(v) else float("nan")
    return out


def poisson_ci(n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if n == 0 else sps.chi2.ppf(alpha / 2, 2 * n) / 2
    hi = sps.chi2.ppf(1 - alpha / 2, 2 * n + 2) / 2
    return float(lo), float(hi)


def group_summary(
    v: np.ndarray,
    u_rand: np.ndarray,
    scale_samples: np.ndarray,
    limit: float | None,
    n_flow: int,
    duration_h: float | None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Point estimates and 95 % intervals for one group of vehicles."""
    v = np.asarray(v, float)
    u = np.nan_to_num(np.asarray(u_rand, float), nan=0.0)
    scale = np.asarray([s for s in scale_samples if s is not None and np.isfinite(s)], float)
    if len(scale) == 0:
        scale = np.ones(1)
    point = speed_statistics(v, limit)
    res = {"n": int(len(v))}
    if len(v) >= 2:
        rng = np.random.default_rng(np.random.SeedSequence([seed, len(v)]))
        boots = {k: np.empty(n_boot) for k in point}
        for b in range(n_boot):
            idx = rng.integers(0, len(v), len(v))
            vb = (v[idx] + rng.normal(0.0, 1.0, len(v)) * u[idx]) * scale[
                rng.integers(0, len(scale))
            ]
            sb = speed_statistics(vb, limit)
            for k in point:
                boots[k][b] = sb[k]
        for k, val in point.items():
            res[k] = val
            res[k + "_ci"] = [
                float(np.percentile(boots[k], 2.5)),
                float(np.percentile(boots[k], 97.5)),
            ]
    else:
        for k, val in point.items():
            res[k] = val
            res[k + "_ci"] = [None, None]
    res["n_counted"] = int(n_flow)
    if duration_h:
        lo, hi = poisson_ci(n_flow)
        res["flow_veh_h"] = n_flow / duration_h
        res["flow_veh_h_ci"] = [lo / duration_h, hi / duration_h]
    return res


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return None if not np.isfinite(o) else round(o, 4)
    return o


def summarize_run(run_dir, n_boot: int = 2000, seed: int = 0) -> dict:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    records = read_records_csv(run_dir / "records.csv")
    decisions = load_decisions(run_dir)
    return summarize(summary, records, decisions, n_boot, seed)


def summarize(
    summary: dict, records: list[dict], decisions: dict | None = None, n_boot=2000, seed=0
) -> dict:
    decisions = decisions or {}
    limit = summary.get("speed_limit_kmh")
    dur = summary.get("duration_s")
    dur_h = dur / 3600.0 if dur else None
    scale = np.array(
        [s for s in summary["uncertainty"]["calibration_scale_samples"] if s is not None], float
    )

    def included(r):
        if r["speed_kmh"] is None:
            return False
        if r["status"] == "valid":
            return decisions.get(r["record_id"]) != "reject"
        if r["status"] == "review":
            return decisions.get(r["record_id"]) == "accept"
        return False

    def counted(r):  # a vehicle crossing for flow purposes
        if r["status"] == "rejected":
            return False
        return decisions.get(r["record_id"]) != "reject"

    inc = [r for r in records if included(r)]
    groups = {"all": (inc, [r for r in records if counted(r)])}
    for d in ("towards", "away"):
        groups[f"direction:{d}"] = (
            [r for r in inc if r["direction"] == d],
            [r for r in records if counted(r) and r["direction"] == d],
        )
    lanes = sorted({r["lane"] for r in records if r["lane"]})
    for ln in lanes:
        groups[f"lane:{ln}"] = (
            [r for r in inc if r["lane"] == ln],
            [r for r in records if counted(r) and r["lane"] == ln],
        )
    out_groups = {}
    for name, (rs, cnt) in groups.items():
        if not cnt and not rs:
            continue
        v = np.array([r["speed_kmh"] for r in rs], float)
        u = np.array([r["u_rand_kmh"] if r["u_rand_kmh"] is not None else 0.0 for r in rs], float)
        out_groups[name] = group_summary(v, u, scale, limit, len(cnt), dur_h, n_boot, seed)
    status_counts = {
        s: sum(r["status"] == s for r in records) for s in ("valid", "review", "rejected")
    }
    reviewed = {
        "accepted": sum(v == "accept" for v in decisions.values()),
        "rejected": sum(v == "reject" for v in decisions.values()),
    }
    pending = sum(1 for r in records if r["status"] == "review" and r["record_id"] not in decisions)
    rep = {
        "site": summary.get("site"),
        "source": summary.get("source"),
        "duration_s": dur,
        "speed_limit_kmh": limit,
        "zone": summary.get("zone"),
        "records": len(records),
        "status_counts": status_counts,
        "review": {**reviewed, "pending": pending},
        "calibration": summary.get("calibration"),
        "calibration_scale_sd": float(np.std(scale)) if len(scale) > 1 else None,
        "groups": out_groups,
        "method": {
            "included": "valid records and review records accepted by a reviewer",
            "flow_counts": "valid and review records (rejected fragments are not counted)",
            "ci": "95% bootstrap over vehicles x per-vehicle random error x calibration scale draws",
            "n_boot": n_boot,
        },
    }
    return _clean(rep)


# ---------------------------------------------------------------------------
# Figures and HTML
# ---------------------------------------------------------------------------


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, metadata={"Software": None})
    return base64.b64encode(buf.getvalue()).decode()


def make_figures(records: list[dict], rep: dict, decisions: dict | None = None) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decisions = decisions or {}
    limit = rep.get("speed_limit_kmh")
    inc = [
        r
        for r in records
        if r["speed_kmh"] is not None
        and (
            (r["status"] == "valid" and decisions.get(r["record_id"]) != "reject")
            or (r["status"] == "review" and decisions.get(r["record_id"]) == "accept")
        )
    ]
    figs = {}
    colors = {"towards": "#2f6f8f", "away": "#c26a2b"}
    # 1. distribution + cumulative
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    bins = np.arange(0, 165, 5)
    for d in ("towards", "away"):
        v = [r["speed_kmh"] for r in inc if r["direction"] == d]
        if v:
            ax[0].hist(v, bins=bins, alpha=0.6, color=colors[d], label=f"{d} (n={len(v)})")
            vs = np.sort(v)
            ax[1].plot(vs, np.arange(1, len(vs) + 1) / len(vs), color=colors[d], label=d)
    allv = np.sort([r["speed_kmh"] for r in inc])
    if len(allv):
        v85 = np.percentile(allv, 85)
        ax[1].axhline(0.85, color="0.5", lw=0.8, ls=":")
        ax[1].axvline(v85, color="0.2", lw=1, ls="--", label=f"V85 all = {v85:.1f}")
    for a in ax:
        if limit:
            a.axvline(limit, color="#b00020", lw=1, label=f"limit {limit:g}")
        a.set_xlabel("speed (km/h)")
        a.legend(fontsize=7)
    ax[0].set_ylabel("vehicles")
    ax[1].set_ylabel("cumulative share")
    fig.tight_layout()
    figs["distribution"] = fig
    # 2. speed over time
    fig, ax = plt.subplots(figsize=(9, 3.0))
    for d in ("towards", "away"):
        rs = [r for r in inc if r["direction"] == d and r["t_mid_s"] is not None]
        if rs:
            t = np.array([r["t_mid_s"] for r in rs]) / 60
            v = np.array([r["speed_kmh"] for r in rs])
            e = 1.96 * np.array([r["u_kmh"] or 0 for r in rs])
            ax.errorbar(t, v, yerr=e, fmt="o", ms=3, lw=0.8, color=colors[d], label=d)
    if limit:
        ax.axhline(limit, color="#b00020", lw=1)
    ax.set_xlabel("time since start (min)")
    ax.set_ylabel("speed (km/h)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    figs["timeline"] = fig
    # 3. flow per interval
    dur = rep.get("duration_s") or 0
    if dur:
        width = 300 if dur >= 1800 else 60
        edges = np.arange(0, dur + width, width)
        fig, ax = plt.subplots(figsize=(9, 2.6))
        cnt = [
            r
            for r in records
            if r["status"] != "rejected"
            and r["t_mid_s"] is not None
            and decisions.get(r["record_id"]) != "reject"
        ]
        bottom = np.zeros(len(edges) - 1)
        for d in ("towards", "away"):
            h, _ = np.histogram([r["t_mid_s"] for r in cnt if r["direction"] == d], bins=edges)
            ax.bar(
                edges[:-1] / 60,
                h * 3600 / width,
                width=width / 60 * 0.9,
                bottom=bottom,
                align="edge",
                color=colors[d],
                label=d,
            )
            bottom += h * 3600 / width
        ax.set_xlabel("time since start (min)")
        ax.set_ylabel(f"flow (veh/h, {width // 60}-min bins)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        figs["flow"] = fig
    return figs


def _fmt(v, nd=1):
    return "–" if v is None else f"{v:.{nd}f}"


def _ci(g, k, nd=1, pct=False):
    v = g.get(k)
    ci = g.get(k + "_ci") or [None, None]
    if v is None:
        return "–"
    f = 100.0 if pct else 1.0
    s = f"{v * f:.{nd}f}"
    if ci[0] is not None:
        s += f" <span class=ci>[{ci[0] * f:.{nd}f}, {ci[1] * f:.{nd}f}]</span>"
    return s


def render_html(rep: dict, fig_b64: dict, title: str = "Speed audit report") -> str:
    limit = rep.get("speed_limit_kmh")
    rows = []
    for name, g in rep["groups"].items():
        cells = [
            html.escape(name),
            str(g["n"]),
            str(g.get("n_counted", "")),
            _ci(g, "mean_kmh"),
            _ci(g, "v85_kmh"),
            _ci(g, "share_over_limit", 1, pct=True) if limit else "–",
            _ci(g, "flow_veh_h", 0) if "flow_veh_h" in g else "–",
        ]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    sc = rep["status_counts"]
    cal = rep.get("calibration") or {}
    imgs = "".join(
        f'<figure><img alt="{html.escape(k)}" src="data:image/png;base64,{b}"></figure>'
        for k, b in fig_b64.items()
    )
    synthetic = (rep.get("source") or "").startswith("synthetic") or "synthetic" in (
        rep.get("site") or ""
    )
    banner = (
        '<p class="note">This report was produced from <b>simulated</b> video. The numbers describe '
        "the simulation, not a real road.</p>"
        if synthetic
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font: 15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; color:#1d2421; background:#f7f8f6; margin:0; padding: 24px 16px; }}
main {{ max-width: 940px; margin: 0 auto; }}
h1 {{ font-size: 1.5rem; margin: 0 0 4px; }} h2 {{ font-size: 1.1rem; margin-top: 28px; }}
.meta {{ color:#5b6660; }} .note {{ background:#fff4d6; border:1px solid #e6cf8b; padding:8px 12px; border-radius:6px; }}
table {{ border-collapse: collapse; width: 100%; background:#fff; }} th, td {{ padding: 6px 8px; border-bottom: 1px solid #e3e6e3; text-align: right; }}
th:first-child, td:first-child {{ text-align:left; }} th {{ font-weight:600; background:#eef1ee; }}
.ci {{ color:#6b756f; font-size: 0.85em; white-space: nowrap; }} .scroll {{ overflow-x:auto; }}
figure {{ margin: 16px 0; }} img {{ max-width:100%; height:auto; background:#fff; border:1px solid #e3e6e3; }}
ul {{ padding-left: 20px; }}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="meta">Site: {html.escape(str(rep.get("site")))} · source: {html.escape(str(rep.get("source")))} ·
duration: {_fmt((rep.get("duration_s") or 0) / 60, 1)} min · zone: {rep.get("zone", {}).get("start_m")}–{rep.get("zone", {}).get("end_m")} m ·
speed limit: {limit if limit is not None else "not set"} km/h</p>
{banner}
<p class="note">Audit statistics for road-safety planning. This is not a certified speed-enforcement
device and its output must not be used to fine or identify anyone. No images of people or number plates are stored.</p>
<h2>Summary</h2>
<div class="scroll"><table>
<tr><th>group</th><th>n (speeds)</th><th>n (counted)</th><th>mean km/h</th><th>V85 km/h</th><th>% over limit</th><th>flow veh/h</th></tr>
{"".join(rows)}
</table></div>
<p class="meta">Brackets: 95 % confidence intervals ({html.escape(rep["method"]["ci"])}; flow: exact Poisson).
Speeds: {html.escape(rep["method"]["included"])}. Flow: {html.escape(rep["method"]["flow_counts"])}.</p>
<h2>Data quality</h2>
<ul>
<li>Records: {rep["records"]} — valid {sc["valid"]}, flagged for review {sc["review"]}, rejected {sc["rejected"]}.</li>
<li>Review: accepted {rep["review"]["accepted"]}, rejected {rep["review"]["rejected"]}, pending {rep["review"]["pending"]} (pending records are excluded from speed statistics).</li>
<li>Calibration: {cal.get("n_points")} points, reprojection RMS {_fmt(cal.get("rms_px"), 2)} px
({_fmt(cal.get("rms_m"), 3)} m); click noise used {_fmt(cal.get("click_sigma_px_used"), 2)} px, survey noise {_fmt(cal.get("ground_sigma_m"), 3)} m;
calibration scale s.d. {_fmt(100 * (rep.get("calibration_scale_sd") or 0), 2)} %.</li>
</ul>
<h2>Figures</h2>
{imgs}
</main></body></html>
"""


def build_report(
    run_dir, out_dir=None, n_boot: int = 2000, seed: int = 0, title: str = "Speed audit report"
) -> dict:
    """Write ``report.json``, ``report.html`` and ``figures/*.png`` for a run directory."""
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = summarize_run(run_dir, n_boot, seed)
    records = read_records_csv(run_dir / "records.csv")
    decisions = load_decisions(run_dir)
    figs = make_figures(records, rep, decisions)
    (out_dir / "figures").mkdir(exist_ok=True)
    b64 = {}
    for name, fig in figs.items():
        fig.savefig(out_dir / "figures" / f"{name}.png", dpi=110, metadata={"Software": None})
        b64[name] = _fig_to_b64(fig)
        plt.close(fig)
    dump_json(rep, out_dir / "report.json")
    (out_dir / "report.html").write_text(render_html(rep, b64, title))
    return rep
