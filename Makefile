# Developer entry points. These mirror the commands documented in the README so
# the repo stays runnable whether or not `make` is installed (on Windows without
# make, copy the command under each target directly).

.PHONY: install install-dev install-dbt test lint format clean         seeds fixture dbt-build dbt-build-fixture

# dbt lives in its own virtual environment (see requirements-dbt.txt for why).
# Scripts/ on Windows, bin/ everywhere else, resolved when the recipe runs so the
# same target works on both machines.
VENV_DBT_PY = $$(if [ -x .venv-dbt/Scripts/python.exe ]; then echo .venv-dbt/Scripts/python.exe; else echo .venv-dbt/bin/python; fi)
DBT = $$(if [ -x ../.venv-dbt/Scripts/dbt.exe ]; then echo ../.venv-dbt/Scripts/dbt.exe; else echo ../.venv-dbt/bin/dbt; fi)

FIXTURE = ../tests/fixtures/lending_club_sample.csv

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,serving,data]"

install-dbt:
	python -m venv .venv-dbt
	$(VENV_DBT_PY) -m pip install --upgrade pip
	$(VENV_DBT_PY) -m pip install -r requirements-dbt.txt

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

# Regenerate the dbt seeds from config.py. Run after changing the feature
# allowlist or the target definition, or `make test` fails on the stale seed.
seeds:
	python -m credit_risk.dbt_seeds

# Regenerate the CI regression fixture. Needs the real export in data/.
fixture:
	python -m credit_risk.fixture

# Full build against the real 2.26M-row export. Local only: the source is a
# frozen 2018Q4 file, so re-running it on every push recomputes a constant.
dbt-build:
	cd dbt && $(DBT) build --profiles-dir .

# What CI runs: the same models and the same contracts, against the committed
# fixture instead of the export.
dbt-build-fixture:
	cd dbt && $(DBT) build --profiles-dir . --target ci --vars '{"raw_accepted_file": "$(FIXTURE)"}'

clean:
	rm -rf .pytest_cache .ruff_cache build *.egg-info
	rm -rf dbt/target dbt/logs dbt/dbt_packages
	find . -type d -name __pycache__ -exec rm -rf {} +
