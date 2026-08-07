"""SHAP feature importance for the primary credit-risk model.

Explains which application-time signals drive the model's risk scores, using SHAP values on a sample of the held-out (test) vintages.
Loads the model saved by train.py if present, otherwise trains it, so the script is self-contained.

Example:
    python -m credit_risk.interpret
    python -m credit_risk.interpret --max-samples 10000
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import shap

from credit_risk import config
from credit_risk.data import load_labeled
from credit_risk.features import engineer_features
from credit_risk.split import time_based_split
from credit_risk.train import MODEL_PATH, train_model

IMPORTANCE_PATH = config.EXPERIMENTS_DIR / "shap_importance.json"


def shap_importance(model, X, max_samples: int = 20000) -> dict:
    """Mean absolute SHAP value per feature (global importance), high to low."""
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=config.RANDOM_SEED)
    values = np.asarray(shap.TreeExplainer(model).shap_values(X))
    if values.ndim == 3:  # a binary classifier may return a per-class slab
        values = values[1] if values.shape[0] == 2 else values[..., 1]
    mean_abs = np.abs(values).mean(axis=0)
    importance = {col: float(v) for col, v in zip(X.columns, mean_abs)}
    return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=20000)
    args = parser.parse_args()

    df = load_labeled(nrows=args.sample)
    train_df, test_df = time_based_split(df)
    X_train = engineer_features(train_df)
    y_train = train_df[config.LABEL_NAME]
    X_test = engineer_features(test_df)

    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else train_model(X_train, y_train)
    importance = shap_importance(model, X_test, max_samples=args.max_samples)

    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    IMPORTANCE_PATH.write_text(json.dumps(importance, indent=2))

    print("SHAP importance (mean |value|), top 15:")
    for i, (col, val) in enumerate(importance.items()):
        if i >= 15:
            break
        print(f"  {col:<24} {val:.4f}")
    print(f"saved -> {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
