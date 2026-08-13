"""Tests for credit_risk.inference.

The key guarantee: scoring one record must match that record's batch prediction.
This catches the classic LightGBM pitfall where a single-row frame's categorical
codes fail to align with the training categories.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk import config
from credit_risk.features import engineer_features
from credit_risk.inference import predict_default_proba, score
from credit_risk.train import train_model


def _raw(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=n) for c in config.RAW_NUMERIC_FEATURES})
    df["term"] = rng.choice([" 36 months", " 60 months"], n)
    df["emp_length"] = rng.choice(["< 1 year", "10+ years", "3 years"], n)
    df["fico_range_low"] = rng.integers(600, 750, n)
    df["fico_range_high"] = df["fico_range_low"] + 4
    df["issue_d"] = rng.choice(["Jan-2015", "Jun-2016", "Dec-2017"], n)
    df["earliest_cr_line"] = rng.choice(["Jan-2000", "Jun-2005"], n)
    df["home_ownership"] = rng.choice(["RENT", "OWN", "MORTGAGE"], n)
    df["verification_status"] = rng.choice(["Verified", "Not Verified"], n)
    df["purpose"] = rng.choice(["credit_card", "car"], n)
    df["addr_state"] = rng.choice(["CA", "TX", "NY"], n)
    return df


def _fitted_model(raw: pd.DataFrame):
    y = pd.Series(np.arange(len(raw)) % 2)
    return train_model(engineer_features(raw), y)


def test_single_record_matches_batch_prediction() -> None:
    raw = _raw()
    model = _fitted_model(raw)
    batch = model.predict_proba(engineer_features(raw))[:, 1]
    single = predict_default_proba(raw.iloc[0].to_dict(), model=model)
    # Category codes must align: the single-row path must equal the batch value.
    assert abs(single - float(batch[0])) < 1e-9


def test_score_returns_valid_decision() -> None:
    raw = _raw()
    model = _fitted_model(raw)
    out = score(raw.iloc[0].to_dict(), model=model, threshold=0.5)
    assert out["decision"] in {"approve", "decline"}
    assert 0.0 <= out["default_probability"] <= 1.0
    assert out["threshold"] == 0.5
