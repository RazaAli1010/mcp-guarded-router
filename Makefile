.PHONY: install check doctor clean kaggle-bundle

# Canonical task definitions. On Windows without GNU make, tasks.ps1 mirrors these targets.

install:
	uv venv .venv
	uv pip install -e ".[dev]"

check:
	ruff check .
	ruff format --check .
	pytest

doctor:
	mcpr doctor

clean:
	rm -rf .pytest_cache .ruff_cache dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

kaggle-bundle:
	@echo "kaggle-bundle: implemented in F6" >&2
	@exit 1
