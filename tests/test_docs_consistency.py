"""The README's numbers must come from files the code wrote (results/)."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

pytestmark = pytest.mark.skipif(
    not (RESULTS / "tables.md").exists(), reason="full experiment results not present"
)


def _section(text: str, start: str, end: str) -> str:
    a = text.index(start)
    return text[a : text.index(end, a)]


def test_readme_result_tables_are_generated_verbatim():
    readme = (ROOT / "README.md").read_text()
    tables = set((RESULTS / "tables.md").read_text().splitlines())
    body = _section(readme, "## 3. Validation on simulated video", "## 4. Placement guidance")
    rows = [ln for ln in body.splitlines() if ln.startswith("| ")]
    assert len(rows) > 50
    missing = [ln for ln in rows if ln not in tables]
    assert not missing, missing[:3]


def test_readme_headline_numbers_match_results():
    readme = (ROOT / "README.md").read_text()
    head = readme[: readme.index("## 1. Why speed")]
    p = json.loads((RESULTS / "e01_baseline.json").read_text())["pooled"]
    e = p["error_valid_kmh"]
    assert f"{p['outcomes']['valid']} of {p['gt_vehicles']}" in head
    for v in (e["bias"], e["rmse"], e["p95_abs"], e["max_abs"]):
        s = f"{abs(v):.2f}"
        assert s in head, s
    v85 = json.loads((RESULTS / "e10_v85.json").read_text())["summary"]["v85_kmh"]
    assert f"{v85['rmse']:.2f}" in head


def test_no_placeholders_left():
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = path.read_text()
        assert "RESULTS-" not in text and "{{" not in text and "TODO" not in text, path
