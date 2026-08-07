"""Tests for credit_risk.calibrate.

Tiny frame with known cashflows verifies the realized-net formula, the LGD/margin estimation, and the analytic profit cutoff.
"""
from __future__ import annotations

import pandas as pd

from credit_risk import config
from credit_risk.calibrate import estimate_lgd_margin, profit_threshold, realized_net


def _econ_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            config.LOAN_AMOUNT_COLUMN: [1000.0, 1000.0],
            config.TOTAL_PAYMENT_COLUMN: [400.0, 1200.0],
            config.RECOVERIES_COLUMN: [100.0, 0.0],
        }
    )


def test_realized_net() -> None:
    # default row: 400 + 100 - 1000 = -500 ; repaid row: 1200 + 0 - 1000 = 200
    assert list(realized_net(_econ_df())) == [-500.0, 200.0]


def test_estimate_lgd_margin() -> None:
    lgd, margin = estimate_lgd_margin(_econ_df(), pd.Series([1, 0]))
    assert lgd == 0.5   # 500 loss / 1000 principal
    assert margin == 0.2  # 200 profit / 1000 principal


def test_profit_threshold_formula() -> None:
    assert profit_threshold(0.65, 0.12) == 0.12 / 0.77
    assert 0.0 < profit_threshold(0.5, 0.2) < 1.0
