"""Hyperparameter tuning for the LightGBM credit-risk model (Optuna).

Searches LightGBM hyperparameters to maximize ROC-AUC on a time-based validation
slice carved out of the training data. The test set is never touched during the
search, so it stays an honest held-out estimate. The best parameters and the test
lift over the current defaults are written to experiments/tuning.json.

Example:
    python -m credit_risk.tune --sample 500000 --trials 40
"""
from __future__ import annotations

import argparse
import json

import optuna
from lightgbm import LGBMClassifier

from credit_risk import config
from credit_risk.data import load_labeled
from credit_risk.evaluate import classification_metrics
from credit_risk.features import engineer_features
from credit_risk.split import time_based_split
from credit_risk.train import train_model

TUNING_PATH = config.EXPERIMENTS_DIR / "tuning.json"


def _scale_pos_weight(y) -> float:
    """Negative/positive ratio used to offset the class imbalance."""
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    return n_neg / max(n_pos, 1)


def _suggest_params(trial: optuna.Trial) -> dict:
    """Sample one LightGBM hyperparameter configuration for a trial."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 700),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def _fit_lgbm(params: dict, X, y) -> LGBMClassifier:
    """Fit a LightGBM classifier with the given params plus the fixed settings."""
    model = LGBMClassifier(
        **params,
        scale_pos_weight=_scale_pos_weight(y),
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=500_000,
                        help="Rows to read for the search (keeps trials quick).")
    parser.add_argument("--trials", type=int, default=40)
    args = parser.parse_args()

    df = load_labeled(nrows=args.sample)
    train_df, test_df = time_based_split(df, test_frac=0.2)
    # Internal temporal validation split from train; the test set stays untouched.
    fit_df, valid_df = time_based_split(train_df, test_frac=0.2)

    X_fit, y_fit = engineer_features(fit_df), fit_df[config.LABEL_NAME]
    X_valid = engineer_features(valid_df)
    y_valid = valid_df[config.LABEL_NAME].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        model = _fit_lgbm(_suggest_params(trial), X_fit, y_fit)
        proba = model.predict_proba(X_valid)[:, 1]
        return classification_metrics(y_valid, proba)["roc_auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
    )
    study.optimize(objective, n_trials=args.trials)

    # Honest comparison: tuned vs current defaults, both scored on the held-out test.
    X_train = engineer_features(train_df)
    y_train = train_df[config.LABEL_NAME]
    X_test = engineer_features(test_df)
    y_test = test_df[config.LABEL_NAME].to_numpy()

    default_auc = classification_metrics(
        y_test, train_model(X_train, y_train).predict_proba(X_test)[:, 1]
    )["roc_auc"]
    tuned_auc = classification_metrics(
        y_test, _fit_lgbm(study.best_params, X_train, y_train).predict_proba(X_test)[:, 1]
    )["roc_auc"]

    result = {
        "best_params": study.best_params,
        "best_valid_roc_auc": float(study.best_value),
        "test_roc_auc_default": float(default_auc),
        "test_roc_auc_tuned": float(tuned_auc),
        "test_roc_auc_lift": float(tuned_auc - default_auc),
        "n_trials": args.trials,
        "sample": args.sample,
    }
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    TUNING_PATH.write_text(json.dumps(result, indent=2))

    print(f"best validation ROC-AUC: {study.best_value:.4f}")
    print("best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(
        f"test ROC-AUC: default {default_auc:.4f} -> tuned {tuned_auc:.4f} "
        f"(lift {tuned_auc - default_auc:+.4f})"
    )
    print(f"saved -> {TUNING_PATH}")


if __name__ == "__main__":
    main()
