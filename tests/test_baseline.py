"""Tests for credit_risk.baseline.

A tiny synthetic frame (with category-dtype columns) confirms the pipeline builds, handles the preprocessing, and produces valid probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk import config
from credit_risk.baseline import build_logistic, train_logistic


def _synthetic(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    numeric = [c for c in config.FEATURE_COLUMNS if c not in config.CATEGORICAL_FEATURES]
    data = {c: rng.normal(size=n) for c in numeric}
    for c in config.CATEGORICAL_FEATURES:
        data[c] = pd.Categorical(rng.choice(["a", "b", "c"], n))
    return pd.DataFrame(data)


def test_build_logistic_has_preprocessing_and_model() -> None:
    pipe = build_logistic()
    assert "preprocess" in pipe.named_steps
    assert "model" in pipe.named_steps


def test_train_logistic_fits_and_predicts_probabilities() -> None:
    X = _synthetic()
    y = np.arange(len(X)) % 2
    model = train_logistic(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
