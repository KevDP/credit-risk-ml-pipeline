"""Tests for credit_risk.tune helpers."""
from __future__ import annotations

import numpy as np

from credit_risk.tune import _scale_pos_weight


def test_scale_pos_weight_is_negative_over_positive_ratio() -> None:
    y = np.array([0, 0, 0, 1])  # 3 negatives, 1 positive
    assert _scale_pos_weight(y) == 3.0


def test_scale_pos_weight_handles_no_positives() -> None:
    y = np.array([0, 0, 0])
    assert _scale_pos_weight(y) == 3.0  # guarded division, no ZeroDivisionError
