"""Tests for credit_risk.features.engineer_features.

Synthetic frame (no dataset needed) that checks the parsing/derivation rules and the leakage guarantee:
the feature matrix is built only from allowlisted, application-time signals.
"""
from __future__ import annotations

import pandas as pd

from credit_risk import config
from credit_risk.features import engineer_features


def _raw_row() -> dict:
    row = {c: 1.0 for c in config.RAW_NUMERIC_FEATURES}
    row.update(
        {
            "term": " 36 months",
            "emp_length": "10+ years",
            "fico_range_low": 700,
            "fico_range_high": 704,
            "issue_d": "Dec-2011",
            "earliest_cr_line": "Dec-2001",
            "home_ownership": "RENT",
            "verification_status": "Verified",
            "purpose": "credit_card",
            "addr_state": "CA",
        }
    )
    return row


def test_output_has_exactly_the_feature_columns() -> None:
    out = engineer_features(pd.DataFrame([_raw_row()]))
    assert list(out.columns) == config.FEATURE_COLUMNS


def test_parsing_and_derivations() -> None:
    out = engineer_features(pd.DataFrame([_raw_row()]))
    assert out.loc[0, "term_months"] == 36
    assert out.loc[0, "emp_length_years"] == 10
    assert out.loc[0, "fico_score"] == 702  # mean of 700 and 704
    # Dec-2001 to Dec-2011 is about 10 years.
    assert 9.9 <= out.loc[0, "credit_history_length"] <= 10.1


def test_categoricals_are_category_dtype() -> None:
    out = engineer_features(pd.DataFrame([_raw_row()]))
    for col in config.CATEGORICAL_FEATURES:
        assert isinstance(out[col].dtype, pd.CategoricalDtype)


def test_feature_source_has_no_leakage_or_risk_columns() -> None:
    # The allowlist must never overlap post-origination or LC risk-model columns.
    forbidden = config.KNOWN_LEAKAGE_COLUMNS | config.LC_RISK_MODEL_COLUMNS
    assert config.FEATURE_SOURCE_COLUMNS.isdisjoint(forbidden)
