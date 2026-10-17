PYTHON := .venv/bin/python
PIP := .venv/bin/pip
CONFIG ?= configs/dev.yaml

.PHONY: help setup data train eval ope serve test lint all clean

help:
	@echo "Available commands:"
	@echo "  make setup   - Create virtualenv and install dependencies"
	@echo "  make data    - Download dataset and generate time splits & features"
	@echo "  make train   - Train Two-Tower retrieval and LightGBM ranker models"
	@echo "  make eval    - Run offline evaluation across all models"
	@echo "  make ope     - Run counterfactual off-policy evaluation study"
	@echo "  make serve   - Launch FastAPI serving application"
	@echo "  make test    - Run pytest test suite (unit, leakage, api)"
	@echo "  make lint    - Run ruff linter and code checks"
	@echo "  make all     - Run end-to-end pipeline (data -> train -> eval -> ope -> report)"
	@echo "  make clean   - Remove cache, temporary data, and build artifacts"

setup:
	python3.12 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) scripts/run_pipeline.py --stage data --config $(CONFIG)

train:
	$(PYTHON) scripts/run_pipeline.py --stage train --config $(CONFIG)

eval:
	$(PYTHON) scripts/run_pipeline.py --stage eval --config $(CONFIG)

ope:
	$(PYTHON) scripts/run_ope_study.py --config $(CONFIG)

serve:
	$(PYTHON) -m uvicorn recsys.serving.app:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest -v tests/

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

all:
	$(PYTHON) scripts/run_pipeline.py --stage all --config $(CONFIG)
	$(PYTHON) scripts/run_ope_study.py --config $(CONFIG)

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ src/*/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
