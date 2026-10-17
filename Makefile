PYTHON := .venv/bin/python
PIP := .venv/bin/pip
CONFIG ?= configs/dev.yaml
NPM ?= npm

.PHONY: help setup data train eval ope serve test lint all clean frontend-install frontend-build frontend-dev

help:
	@echo "Available commands:"
	@echo "  make setup            - Create virtualenv and install dependencies"
	@echo "  make frontend-install - Install frontend Node dependencies"
	@echo "  make frontend-build   - Build production React frontend into frontend/dist"
	@echo "  make frontend-dev     - Start Vite hot-reloading frontend development server"
	@echo "  make serve            - Launch production FastAPI server & UI on http://localhost:8000"
	@echo "  make test             - Run pytest test suite (unit, leakage, and full API integration)"
	@echo "  make lint             - Run ruff linter and code checks"
	@echo "  make all              - Run end-to-end pipeline (data -> train -> eval -> ope -> report)"
	@echo "  make clean            - Remove cache, temporary data, and build artifacts"

setup:
	python3.12 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@if [ -d frontend ]; then cd frontend && $(NPM) install; fi

frontend-install:
	cd frontend && $(NPM) install

frontend-build:
	cd frontend && $(NPM) run build

frontend-dev:
	cd frontend && $(NPM) run dev

data:
	$(PYTHON) scripts/run_pipeline.py --stage data --config $(CONFIG)

train:
	$(PYTHON) scripts/run_pipeline.py --stage train --config $(CONFIG)

eval:
	$(PYTHON) scripts/run_pipeline.py --stage eval --config $(CONFIG)

ope:
	$(PYTHON) scripts/run_ope_study.py --config $(CONFIG)

serve:
	@if [ ! -d "frontend/dist" ]; then echo "Building frontend..."; cd frontend && $(NPM) run build && cd ..; fi
	PYTHONPATH=src $(PYTHON) -m uvicorn recsys.serving.app:app --host 0.0.0.0 --port 8000 --reload

test:
	PYTHONPATH=src $(PYTHON) -m pytest -v tests/

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

all:
	$(PYTHON) scripts/run_pipeline.py --stage all --config $(CONFIG)
	$(PYTHON) scripts/run_ope_study.py --config $(CONFIG)

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ src/*/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
