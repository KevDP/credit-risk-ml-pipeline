# Developer entry points. These mirror the commands documented in the README so
# the repo stays runnable whether or not `make` is installed (on Windows without
# make, copy the command under each target directly).

.PHONY: install install-dev test lint format clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,serving]"

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
