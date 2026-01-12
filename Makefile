.PHONY: help install install-dev dev type-check format format-check import-sort import-sort-check test test-verbose lint check fix clean

# Variables
VENV_BIN := .venv/bin
UVICORN := $(VENV_BIN)/uvicorn
PIP := $(VENV_BIN)/pip
PYTEST := $(VENV_BIN)/pytest
APP := main:app
PORT := 8080
HOST := localhost

help:
	@echo "Available targets:"
	@echo "  make install          - Install dependencies"
	@echo "  make install-dev      - Install dependencies with dev extras"
	@echo "  make dev              - Run the server in development mode"
	@echo "  make type-check       - Run mypy type checking"
	@echo "  make format           - Auto-format code with black"
	@echo "  make format-check     - Check code formatting (no changes)"
	@echo "  make import-sort      - Auto-sort imports with isort"
	@echo "  make import-sort-check - Check import sorting (no changes)"
	@echo "  make test             - Run tests with pytest"
	@echo "  make test-verbose     - Run tests with verbose output"
	@echo "  make lint             - Run all linting checks (type, format, imports)"
	@echo "  make check            - Run all checks (lint + test)"
	@echo "  make fix              - Auto-fix format and imports"
	@echo "  make clean            - Clean cache and temporary files"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install black isort mypy pytest

dev:
	$(UVICORN) $(APP) --host $(HOST) --port $(PORT) --reload

type-check:
	mypy api/ services/ models/ data_layer/ telemetrics/ tests/ *.py --ignore-missing-imports

format:
	black api/ services/ models/ data_layer/ telemetrics/ tests/ *.py

format-check:
	black api/ services/ models/ data_layer/ telemetrics/ tests/ *.py --check

import-sort:
	isort api/ services/ models/ data_layer/ telemetrics/ tests/ *.py

import-sort-check:
	isort api/ services/ models/ data_layer/ telemetrics/ tests/ *.py --check

test:
	$(PYTEST) tests/

test-verbose:
	$(PYTEST) tests/ -v

lint: type-check format-check import-sort-check

check: lint test

fix: format import-sort

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
