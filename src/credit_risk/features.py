"""Feature engineering for the credit-risk pipeline.

Turns the raw labeled frame into the model-ready feature matrix defined by config.FEATURE_COLUMNS.
Only application-time columns are used (the allowlist in config), so no post-origination information can leak into the model.

Design:
    - engineer_features is the single transform from raw columns to model input.
      The notebook, training, and serving all call it, so the feature definition never diverges across them.
    - term, emp_length, the FICO range, and the credit-history length are parsed into numeric features here;
      the raw source columns never reach the model.
    - Categorical features are returned as the pandas 'category' dtype, which LightGBM consumes natively
      (no one-hot explosion on high-cardinality columns like addr_state).
"""
from __future__ import annotations

import pandas as pd

from credit_risk import config

# Maps Lending Club's emp_length strings to an ordinal number of years.
_EMP_LENGTH_MAP = {
    "< 1 year": 0.0,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}

# Lending Club dates look like "Dec-2011".
_MONTH_YEAR_FORMAT = "%b-%Y"


def _parse_term(term: pd.Series) -> pd.Series:
    """' 36 months' -> 36.0. Non-parseable values become NaN."""
    return term.str.extract(r"(\d+)", expand=False).astype("float")


def _parse_emp_length(emp_length: pd.Series) -> pd.Series:
    """'< 1 year'..'10+ years' -> 0..10. Unknown/NaN stays NaN."""
    return emp_length.map(_EMP_LENGTH_MAP)


def parse_month_year(dates: pd.Series) -> pd.Series:
    """'Dec-2011' -> Timestamp. Unparseable values become NaT."""
    return pd.to_datetime(dates, format=_MONTH_YEAR_FORMAT, errors="coerce")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model-ready feature matrix (config.FEATURE_COLUMNS).

    Selects the allowlisted raw columns, parses term / emp_length / dates into numeric features,
    derives fico_score and credit_history_length, and returns the categorical columns as the 'category' dtype.
    Returns a new frame; the input is not modified.
    """
    out = df[config.RAW_NUMERIC_FEATURES].copy()

    out["term_months"] = _parse_term(df[config.TERM_COLUMN])
    out["emp_length_years"] = _parse_emp_length(df[config.EMP_LENGTH_COLUMN])
    out["fico_score"] = (df[config.FICO_LOW_COLUMN] + df[config.FICO_HIGH_COLUMN]) / 2.0

    issued = parse_month_year(df[config.ISSUE_DATE_COLUMN])
    opened = parse_month_year(df[config.EARLIEST_CREDIT_LINE_COLUMN])
    out["credit_history_length"] = (issued - opened).dt.days / 365.25

    for col in config.CATEGORICAL_FEATURES:
        out[col] = df[col].astype("category")

    return out[config.FEATURE_COLUMNS]
