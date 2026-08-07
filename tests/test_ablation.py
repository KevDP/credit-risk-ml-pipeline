"""Test for credit_risk.ablation.augment_with_lc_risk."""

from __future__ import annotations

import pandas as pd

from credit_risk import config
from credit_risk.ablation import RISK_CATEGORICAL, RISK_NUMERIC, augment_with_lc_risk


def test_augment_adds_lc_risk_columns_without_mutating_input() -> None:
    base_col = config.FEATURE_COLUMNS[0]
    X = pd.DataFrame({base_col: [1.0, 2.0]})
    df = pd.DataFrame(
        {"int_rate": [12.5, 8.0], "grade": ["B", "A"], "sub_grade": ["B3", "A1"]}
    )
    out = augment_with_lc_risk(X, df)

    for col in RISK_NUMERIC + RISK_CATEGORICAL:
        assert col in out.columns
    assert isinstance(out["grade"].dtype, pd.CategoricalDtype)
    assert base_col in out.columns  # original feature preserved
    assert "int_rate" not in X.columns  # input untouched
