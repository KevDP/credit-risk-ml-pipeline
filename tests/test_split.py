"""Tests for credit_risk.split.time_based_split.

Verifies the split is genuinely temporal: the test set holds the most recent
loans and never overlaps the training period.
"""
from __future__ import annotations

import pandas as pd

from credit_risk.config import ISSUE_DATE_COLUMN
from credit_risk.features import parse_month_year
from credit_risk.split import temporal_folds, time_based_split


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ISSUE_DATE_COLUMN: ["Jan-2015", "Jan-2016", "Jan-2017", "Jan-2018"],
            "x": [1, 2, 3, 4],
        }
    )


def test_test_set_is_the_most_recent() -> None:
    train, test = time_based_split(_df(), test_frac=0.5)
    assert set(train["x"]) == {1, 2}
    assert set(test["x"]) == {3, 4}


def test_no_temporal_overlap() -> None:
    train, test = time_based_split(_df(), test_frac=0.25)
    latest_train = parse_month_year(train[ISSUE_DATE_COLUMN]).max()
    earliest_test = parse_month_year(test[ISSUE_DATE_COLUMN]).min()
    assert latest_train < earliest_test


def test_temporal_folds_expand_and_stay_ordered() -> None:
    df = pd.DataFrame(
        {
            ISSUE_DATE_COLUMN: ["Jan-2015", "Jan-2016", "Jan-2017", "Jan-2018", "Jan-2019"],
            "x": [1, 2, 3, 4, 5],
        }
    )
    folds = list(temporal_folds(df, min_train_years=2))
    # Test years are every year after the first two: 2017, 2018, 2019.
    assert [year for year, _, _ in folds] == [2017, 2018, 2019]
    _, first_train, first_test = folds[0]
    assert set(first_train["x"]) == {1, 2}  # trains on 2015-2016
    assert set(first_test["x"]) == {3}      # tests 2017
