"""Time-based split and expanding-window temporal folds.

Splitting by issue date (not randomly) mirrors production: the model trains on past vintages and is scored on later ones.
A random split leaks future information across the boundary and overstates performance, exactly the failure mode a credit model must avoid.
temporal_folds extends the same idea to a rolling backtest so stability across vintages can be measured, not just asserted.
"""
from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from credit_risk import config
from credit_risk.features import parse_month_year


def time_based_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, test) by issue date.

    The most recent `test_frac` of loans (by issue_d) become the test set; the
    earlier loans are training. Returns two new frames; the input is not changed.
    """
    issued = parse_month_year(df[config.ISSUE_DATE_COLUMN])
    cutoff = issued.quantile(1.0 - test_frac)
    train = df.loc[issued < cutoff].copy()
    test = df.loc[issued >= cutoff].copy()
    return train, test


def temporal_folds(
    df: pd.DataFrame, min_train_years: int = 3
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (test_year, train_df, test_df) expanding-window folds by issue year.

    For each test year after the first `min_train_years`, train on all earlier
    years. Simulates periodic retraining and exposes how performance holds up
    vintage by vintage.
    """
    issued = parse_month_year(df[config.ISSUE_DATE_COLUMN])
    years = issued.dt.year
    unique_years = sorted(int(y) for y in years.dropna().unique())
    for test_year in unique_years[min_train_years:]:
        yield test_year, df.loc[years < test_year], df.loc[years == test_year]
