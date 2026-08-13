"""Inference for the credit-risk model.

Loads a trained model and scores a single loan application. Pure and free of any
web or cloud dependency, so the same code backs the notebook, the tests, and the
Lambda handler. A record is a mapping of the raw application-time fields; the same
engineer_features transform used in training builds the feature row, so the model
always sees identically shaped input.
"""
from __future__ import annotations

import io
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from credit_risk import config
from credit_risk.features import engineer_features

MODEL_PATH = config.MODELS_DIR / "model.joblib"

# Illustrative operating point on the default probability: approve when
# P(default) < threshold. Near the cost-calibrated cutoff from the threshold
# study; override per call. A production deployment should set it on calibrated
# scores (see the model card).
DEFAULT_THRESHOLD = 0.33


def _load_from_s3(uri: str):
    """Stream a joblib model from an s3://bucket/key URI (used in Lambda)."""
    import boto3  # imported lazily so local use needs no AWS SDK

    bucket, key = uri.removeprefix("s3://").split("/", 1)
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    return joblib.load(io.BytesIO(body))


@lru_cache(maxsize=1)
def load_model(path: str | Path = MODEL_PATH):
    """Load the trained model once per process (cached for warm reuse).

    In AWS Lambda, MODEL_S3_URI (s3://bucket/key) points to the artifact in S3 and
    the model is streamed from there; otherwise it loads from the local file.
    """
    s3_uri = os.environ.get("MODEL_S3_URI")
    if s3_uri:
        return _load_from_s3(s3_uri)
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first: python -m credit_risk.train"
        )
    return joblib.load(model_path)


def _record_to_frame(record: Mapping[str, Any]) -> pd.DataFrame:
    """Build a one-row raw frame with every feature-source column (missing -> None)."""
    row = {col: record.get(col) for col in config.FEATURE_SOURCE_COLUMNS}
    return pd.DataFrame([row])


def predict_default_proba(record: Mapping[str, Any], model: Any = None) -> float:
    """Probability that the loan defaults, in [0, 1]."""
    model = model if model is not None else load_model()
    features = engineer_features(_record_to_frame(record))
    return float(model.predict_proba(features)[0, 1])


def score(
    record: Mapping[str, Any], model: Any = None, threshold: float = DEFAULT_THRESHOLD
) -> dict:
    """Score one application: default probability, decision, and threshold used."""
    proba = predict_default_proba(record, model=model)
    return {
        "default_probability": proba,
        "decision": "decline" if proba >= threshold else "approve",
        "threshold": threshold,
    }
