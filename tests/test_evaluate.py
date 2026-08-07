"""Tests for credit_risk.evaluate.

Small deterministic arrays check the ranking metrics and the cost accounting that drives threshold selection.
"""
from __future__ import annotations

import numpy as np

from credit_risk.evaluate import (
    brier_score,
    classification_metrics,
    cost_at_threshold,
    optimal_threshold,
    policy_outcome,
    realized_profit_at_threshold,
)


def test_perfect_ranking_gives_auc_one() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y_true, y_proba)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["positive_rate"] == 0.5
    assert m["n"] == 4


def test_cost_counts_false_negatives_and_positives() -> None:
    # row 0: true default scored low  -> missed default (false negative)
    # row 1: true repay scored high   -> declined good loan (false positive)
    y_true = np.array([1, 0])
    y_proba = np.array([0.1, 0.9])
    cost = cost_at_threshold(y_true, y_proba, threshold=0.5, cost_fn=1.0, cost_fp=0.2)
    assert cost == 1.2


def test_optimal_threshold_finds_zero_cost_separation() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, cost = optimal_threshold(y_true, y_proba, cost_fn=1.0, cost_fp=1.0)
    assert cost == 0.0
    assert 0.2 < threshold <= 0.8


def test_realized_profit_and_policy_outcome() -> None:
    # Approve the loan scoring 0.1 (below 0.5), decline the one scoring 0.9.
    y_true = np.array([0, 1])
    y_proba = np.array([0.1, 0.9])
    net = np.array([200.0, -500.0])
    assert realized_profit_at_threshold(y_proba, net, 0.5) == 200.0
    out = policy_outcome(y_true, y_proba, net, 0.5)
    assert out["approval_rate"] == 0.5
    assert out["realized_profit"] == 200.0
    assert out["book_default_rate"] == 0.0  # the approved loan did not default


def test_brier_score_is_zero_for_perfect_probabilities() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y_true, y_proba) == 0.0
