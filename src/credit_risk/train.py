"""Train and evaluate the baseline credit-risk model.

End-to-end entry point: load the labeled data, split it by issue date, engineer features, fit a LightGBM classifier,
evaluate on the held-out (future) vintages, pick a cost-optimal decision threshold, and save the model and metrics.

The default cost ratio (a false negative costs --cost-fn, a false positive costs --cost-fp) is a placeholder:
approving a defaulter loses far more than declining a good borrower. Tune it once the business numbers are set;
ROC-AUC and PR-AUC do not depend on it.

Examples:
    python -m credit_risk.train
    python -m credit_risk.train --sample 200000 --cost-fn 1.0 --cost-fp 0.2
"""
from __future__ import annotations

import argparse
import json

import joblib
from lightgbm import LGBMClassifier

from credit_risk import config
from credit_risk.data import load_labeled
from credit_risk.evaluate import classification_metrics, optimal_threshold
from credit_risk.features import engineer_features
from credit_risk.split import time_based_split

MODEL_PATH = config.MODELS_DIR / "model.joblib"
METRICS_PATH = config.EXPERIMENTS_DIR / "metrics.json"


def train_model(X_train, y_train) -> LGBMClassifier:
    """Fit a LightGBM classifier, weighting the minority (default) class.

    scale_pos_weight offsets the ~20% default rate. Categorical columns arrive as the 'category' dtype, which LightGBM consumes natively.
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None,
                        help="Read only the first N rows (fast dev runs).")
    parser.add_argument("--test-frac", type=float, default=0.2,
                        help="Fraction of the most recent loans used for test.")
    parser.add_argument("--cost-fn", type=float, default=1.0,
                        help="Cost of a false negative (approved default).")
    parser.add_argument("--cost-fp", type=float, default=0.2,
                        help="Cost of a false positive (declined good loan).")
    args = parser.parse_args()

    df = load_labeled(nrows=args.sample)
    train_df, test_df = time_based_split(df, test_frac=args.test_frac)

    X_train = engineer_features(train_df)
    y_train = train_df[config.LABEL_NAME]
    X_test = engineer_features(test_df)
    y_test = test_df[config.LABEL_NAME]

    print(f"train: {len(train_df):,} rows | test: {len(test_df):,} rows")
    model = train_model(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, proba)
    threshold, cost = optimal_threshold(y_test, proba, args.cost_fn, args.cost_fp)
    metrics.update(
        {
            "threshold": threshold,
            "cost_at_threshold": cost,
            "cost_fn": args.cost_fn,
            "cost_fp": args.cost_fp,
        }
    )

    print("ROC-AUC: {roc_auc:.4f}  PR-AUC: {pr_auc:.4f}".format(**metrics))
    print(f"cost-optimal threshold: {threshold:.3f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"saved model   -> {MODEL_PATH}")
    print(f"saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
