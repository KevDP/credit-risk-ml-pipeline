"""Contract tests between the dbt feature mart and the Python feature path.

Moving feature construction into SQL buys versioned, testable transformations,
but it costs something the repo was careful about: `engineer_features` used to be
the single definition shared by training and serving. dbt cannot run inside the
Lambda, so serving keeps the Python transform and the mart becomes a second
implementation of the same logic. Two implementations of one transformation is
the definition of training/serving skew.

These tests turn that risk into an invariant the build checks:

    - test_seeds_match_config guards config -> SQL. Editing config.py without
      running `python -m credit_risk.dbt_seeds` leaves the SQL allowlist
      describing a different model than the Python one.

    - test_mart_matches_engineer_features guards SQL -> Python. For the same
      loans, every feature value must agree, so SQL cannot quietly drift from
      what the Lambda computes.

The second test is skipped when the warehouse has not been built, so a clean
checkout without the raw Lending Club file still runs green.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from credit_risk import config, dbt_seeds
from credit_risk.features import engineer_features

# Enough rows to exercise every branch (both terms, the full emp_length map,
# null handling) while keeping the test well under a second.
CONTRACT_SAMPLE_SIZE = 5_000

WAREHOUSE = config.DATA_DIR / "warehouse.duckdb"


def _read_seed(path: Path) -> list[tuple[str, ...]]:
    """Read a committed seed CSV as rows of strings, header excluded."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return [tuple(row) for row in rows[1:]]


def test_seeds_match_config() -> None:
    """The committed seeds must be exactly what config.py produces today."""
    expected_allowlist = [(name, role) for name, role in dbt_seeds.build_allowlist_rows()]
    expected_labels = [(status, str(label)) for status, label in dbt_seeds.build_label_rows()]

    assert _read_seed(dbt_seeds.ALLOWLIST_SEED) == expected_allowlist, (
        "dbt/seeds/application_time_allowlist.csv is stale. "
        "Run: python -m credit_risk.dbt_seeds"
    )
    assert _read_seed(dbt_seeds.LABEL_SEED) == expected_labels, (
        "dbt/seeds/loan_status_labels.csv is stale. Run: python -m credit_risk.dbt_seeds"
    )


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Make the two paths comparable without hiding real differences.

    Only representation is normalized: categoricals become a nullable string
    dtype (DuckDB returns objects, pandas returns the category dtype) and the
    column order is fixed. Values themselves are left alone.
    """
    out = frame[config.FEATURE_COLUMNS].copy()
    for column in config.CATEGORICAL_FEATURES:
        out[column] = out[column].astype("string")
    return out.reset_index(drop=True)


@pytest.mark.skipif(
    not WAREHOUSE.exists(),
    reason="warehouse not built; run `make dbt-build` first",
)
def test_mart_matches_engineer_features() -> None:
    """The mart and engineer_features must agree on every feature value."""
    duckdb = pytest.importorskip("duckdb")

    with duckdb.connect(str(WAREHOUSE), read_only=True) as connection:
        # Both sides start from the same rows: staging is a faithful copy of the
        # application-time source columns, so feeding it to engineer_features
        # compares the two transformations and nothing else.
        raw = connection.execute(
            f"select * from stg_loans order by id limit {CONTRACT_SAMPLE_SIZE}"
        ).df()
        actual = connection.execute(
            f"""
            select * from fct_loan_features
            where id in (select id from stg_loans order by id limit {CONTRACT_SAMPLE_SIZE})
            order by id
            """
        ).df()

    assert len(raw) == CONTRACT_SAMPLE_SIZE, "staging model is smaller than the sample size"
    assert len(actual) == len(raw), "mart and staging disagree on row count for the same ids"

    expected = engineer_features(raw)

    pd.testing.assert_frame_equal(
        _normalize(actual),
        _normalize(expected),
        check_dtype=False,
        rtol=1e-9,
        obj="dbt feature mart vs engineer_features",
    )
