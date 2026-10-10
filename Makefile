# roadspeed - common tasks. Everything runs inside ./.venv (no system packages needed).
PY ?= .venv/bin/python
PIP_FLAGS ?= --retries 10 --timeout 60

.PHONY: setup test lint format reproduce reproduce-quick demo clean

setup:
	test -d .venv || python3 -m venv .venv
	$(PY) -m pip install $(PIP_FLAGS) --upgrade pip
	$(PY) -m pip install $(PIP_FLAGS) -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests experiments
	$(PY) -m ruff format --check src tests experiments

format:
	$(PY) -m ruff format src tests experiments
	$(PY) -m ruff check --fix src tests experiments

# Full synthetic validation: writes results/ and figures/ (about an hour on 10 cores).
reproduce:
	$(PY) experiments/run_all.py
	$(PY) experiments/demo_report.py

# Small version for a quick check: writes results/quick/ and figures/quick/ (a few minutes).
reproduce-quick:
	$(PY) experiments/run_all.py --quick
	$(PY) experiments/demo_report.py --quick

# Render a short simulated clip, measure it and build the report in work/demo/.
demo:
	.venv/bin/roadspeed synth --out work/demo/sim --preset baseline --seed 7 --duration 60
	.venv/bin/roadspeed run work/demo/sim/video.mp4 --config work/demo/sim/site.json --out work/demo/run --review-sheets
	.venv/bin/roadspeed report work/demo/run --title "Speed audit report (simulated demo)"

clean:
	rm -rf work .pytest_cache .ruff_cache results/quick figures/quick
