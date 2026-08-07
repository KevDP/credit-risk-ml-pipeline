"""Tests for credit_risk.data.add_label.

Uses a tiny in-memory frame (no dataset download needed) to verify the two guarantees that matter:
correct label mapping, and dropping loans whose outcome is not yet known.
"""
from __future__ import annotations

import pandas as pd

from credit_risk.config import LABEL_NAME
from credit_risk.data import add_label


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_status": ["Charged Off", "Fully Paid", "Current", "Default"],
            "loan_amnt": [1000, 2000, 3000, 4000],
        }
    )


def test_labels_are_mapped_correctly() -> None:
    out = add_label(_sample())
    # Charged Off and Default -> 1; Fully Paid -> 0; Current is dropped.
    assert list(out[LABEL_NAME]) == [1, 0, 1]


def test_in_progress_rows_are_dropped() -> None:
    out = add_label(_sample())
    assert len(out) == 3  # the "Current" row is gone
    assert "Current" not in set(out["loan_status"])


def test_input_frame_is_not_mutated() -> None:
    df = _sample()
    add_label(df)
    assert LABEL_NAME not in df.columns
