"""Tests for the FastAPI serving app.

A tiny in-memory model is injected via an autouse dependency override, so the
tests need neither the trained artifact nor a running server.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_risk import config
from credit_risk.features import engineer_features
from credit_risk.train import train_model
from serving.app import app, get_model

client = TestClient(app)

_EXAMPLE = {
    "loan_amnt": 10000, "annual_inc": 60000, "dti": 15.0, "open_acc": 8,
    "pub_rec": 0, "revol_bal": 5000, "revol_util": 30.0, "total_acc": 20,
    "delinq_2yrs": 0, "inq_last_6mths": 1, "mort_acc": 1, "pub_rec_bankruptcies": 0,
    "fico_range_low": 700, "fico_range_high": 704,
    "term": " 36 months", "emp_length": "5 years",
    "issue_d": "Jan-2018", "earliest_cr_line": "Jan-2005",
    "home_ownership": "RENT", "verification_status": "Verified",
    "purpose": "credit_card", "addr_state": "CA",
}


def _tiny_model():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({c: rng.normal(size=n) for c in config.RAW_NUMERIC_FEATURES})
    df["term"] = rng.choice([" 36 months", " 60 months"], n)
    df["emp_length"] = rng.choice(["< 1 year", "10+ years", "5 years"], n)
    df["fico_range_low"] = rng.integers(600, 750, n)
    df["fico_range_high"] = df["fico_range_low"] + 4
    df["issue_d"] = rng.choice(["Jan-2015", "Jan-2018"], n)
    df["earliest_cr_line"] = rng.choice(["Jan-2000", "Jan-2005"], n)
    df["home_ownership"] = rng.choice(["RENT", "OWN", "MORTGAGE"], n)
    df["verification_status"] = rng.choice(["Verified", "Not Verified"], n)
    df["purpose"] = rng.choice(["credit_card", "car"], n)
    df["addr_state"] = rng.choice(["CA", "TX", "NY"], n)
    return train_model(engineer_features(df), pd.Series(np.arange(n) % 2))


@pytest.fixture(autouse=True)
def _use_tiny_model():
    app.dependency_overrides[get_model] = _tiny_model
    yield
    app.dependency_overrides.clear()


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_score() -> None:
    resp = client.post("/predict", json=_EXAMPLE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in {"approve", "decline"}
    assert 0.0 <= body["default_probability"] <= 1.0


def test_predict_rejects_malformed_request() -> None:
    bad = dict(_EXAMPLE)
    del bad["loan_amnt"]
    assert client.post("/predict", json=bad).status_code == 422
