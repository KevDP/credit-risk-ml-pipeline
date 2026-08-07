"""Ablation: how much does Lending Club's own risk score add?

The primary model deliberately excludes LC's risk-model outputs (grade, sub_grade, int_rate)
to learn an independent risk view. This ablation adds them back and measures the lift, which puts the primary model's ROC-AUC in context.

Example:
    python -m credit_risk.ablation
    python -m credit_risk.ablation --sample 300000
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from credit_risk import config
from credit_risk.data import add_label, load_raw
from credit_risk.evaluate import classification_metrics
from credit_risk.features import engineer_features
from credit_risk.split import time_based_split
from credit_risk.train import train_model

RISK_NUMERIC = ["int_rate"]
RISK_CATEGORICAL = ["grade", "sub_grade"]

RESULT_PATH = config.EXPERIMENTS_DIR / "ablation.json"


def augment_with_lc_risk(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Append LC's risk-model columns to a feature matrix. Input is not mutated."""
    out = X.copy()
    for col in RISK_NUMERIC:
        out[col] = df[col].astype("float")
    for col in RISK_CATEGORICAL:
        out[col] = df[col].astype("category")
    return out


def _load(nrows: int | None) -> pd.DataFrame:
    columns = sorted(
        config.FEATURE_SOURCE_COLUMNS
        | set(RISK_NUMERIC)
        | set(RISK_CATEGORICAL)
        | {config.TARGET_COLUMN}
    )
    return add_label(load_raw(nrows=nrows, columns=columns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    df = _load(args.sample)
    train_df, test_df = time_based_split(df, test_frac=args.test_frac)
    y_train = train_df[config.LABEL_NAME]
    y_test = test_df[config.LABEL_NAME].to_numpy()

    X_train = engineer_features(train_df)
    X_test = engineer_features(test_df)
    primary = train_model(X_train, y_train)
    primary_metrics = classification_metrics(y_test, primary.predict_proba(X_test)[:, 1])

    Xa_train = augment_with_lc_risk(X_train, train_df)
    Xa_test = augment_with_lc_risk(X_test, test_df)
    augmented = train_model(Xa_train, y_train)
    augmented_metrics = classification_metrics(y_test, augmented.predict_proba(Xa_test)[:, 1])

    result = {
        "primary": primary_metrics,
        "with_lc_risk": augmented_metrics,
        "roc_auc_lift": augmented_metrics["roc_auc"] - primary_metrics["roc_auc"],
        "pr_auc_lift": augmented_metrics["pr_auc"] - primary_metrics["pr_auc"],
    }

    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))

    print(f"primary    ROC-AUC {primary_metrics['roc_auc']:.4f}  PR-AUC {primary_metrics['pr_auc']:.4f}")
    print(f"+ LC risk  ROC-AUC {augmented_metrics['roc_auc']:.4f}  PR-AUC {augmented_metrics['pr_auc']:.4f}")
    print(f"lift       ROC-AUC +{result['roc_auc_lift']:.4f}  PR-AUC +{result['pr_auc_lift']:.4f}")
    print(f"saved -> {RESULT_PATH}")


if __name__ == "__main__":
    main()
