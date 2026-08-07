"""Evaluation metrics and cost-based threshold selection.

Ranking metrics (ROC-AUC, PR-AUC) are threshold-free: they measure how well the model orders borrowers by risk.
The decision threshold is a separate, business choice that minimizes expected cost given the asymmetric penalties of approving a
defaulter (false negative) versus declining a good borrower (false positive).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def classification_metrics(y_true: ArrayLike, y_proba: ArrayLike) -> dict:
    """Threshold-free ranking metrics plus the base rate and sample size."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "positive_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def brier_score(y_true: ArrayLike, y_proba: ArrayLike) -> float:
    """Mean squared error of the predicted probabilities. Lower is better
    calibrated; a cost-based threshold is only trustworthy if this is low."""
    return float(brier_score_loss(np.asarray(y_true), np.asarray(y_proba)))


def cost_at_threshold(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> float:
    """Total cost when loans scoring >= threshold are declined as risky.

    A false negative (predict repay, actually defaults) costs `cost_fn`; a false
    positive (predict default, actually repays) costs `cost_fp`.
    """
    y_true = np.asarray(y_true)
    predicted_default = np.asarray(y_proba) >= threshold
    false_neg = int(np.sum(~predicted_default & (y_true == 1)))
    false_pos = int(np.sum(predicted_default & (y_true == 0)))
    return cost_fn * false_neg + cost_fp * false_pos


def optimal_threshold(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    cost_fn: float,
    cost_fp: float,
    n_steps: int = 101,
) -> tuple[float, float]:
    """Scan thresholds in [0, 1]; return the (threshold, cost) with lowest cost."""
    thresholds = np.linspace(0.0, 1.0, n_steps)
    costs = [
        cost_at_threshold(y_true, y_proba, t, cost_fn, cost_fp) for t in thresholds
    ]
    best = int(np.argmin(costs))
    return float(thresholds[best]), float(costs[best])


def realized_profit_at_threshold(
    y_proba: ArrayLike, net: ArrayLike, threshold: float
) -> float:
    """Realized dollars from approving loans that score below `threshold`."""
    approved = np.asarray(y_proba) < threshold
    return float(np.asarray(net)[approved].sum())


def policy_outcome(
    y_true: ArrayLike, y_proba: ArrayLike, net: ArrayLike, threshold: float
) -> dict:
    """Business outcome of the approve-if-below-threshold policy: approval rate,
    realized dollars, and the default rate of the approved book."""
    y_true = np.asarray(y_true)
    approved = np.asarray(y_proba) < threshold
    net = np.asarray(net)
    n_approved = int(approved.sum())
    return {
        "threshold": float(threshold),
        "approval_rate": float(approved.mean()),
        "realized_profit": float(net[approved].sum()),
        "book_default_rate": float(y_true[approved].mean()) if n_approved else 0.0,
    }
