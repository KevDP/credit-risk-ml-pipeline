"""Phase 1 robustness study: model comparison, calibration, temporal stability.

Three checks that back the modeling choices:

  1. Baseline vs GBM: does LightGBM beat a logistic-regression scorecard by enough to justify the added complexity?

  2. Calibration: are the GBM's probabilities reliable enough for a cost-based threshold?
     Isotonic calibration on a held-out slice, Brier before/after.

  3. Temporal stability: expanding-window ROC-AUC per vintage year.

Results go to experiments/validation.json.

Example:
    python -m credit_risk.validate
    python -m credit_risk.validate --sample 400000
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.isotonic import IsotonicRegression

from credit_risk import config
from credit_risk.baseline import train_logistic
from credit_risk.data import load_labeled
from credit_risk.evaluate import brier_score, classification_metrics
from credit_risk.features import engineer_features
from credit_risk.split import temporal_folds, time_based_split
from credit_risk.train import train_model

VALIDATION_PATH = config.EXPERIMENTS_DIR / "validation.json"
# Cap per-fold training size so the rolling backtest stays quick without meaningfully changing the stability picture.
FOLD_TRAIN_CAP = 300_000


def _scored(y_true, y_proba) -> dict:
    return {**classification_metrics(y_true, y_proba), "brier": brier_score(y_true, y_proba)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    df = load_labeled(nrows=args.sample)
    train_df, test_df = time_based_split(df, test_frac=0.2)
    X_train = engineer_features(train_df)
    y_train = train_df[config.LABEL_NAME]
    X_test = engineer_features(test_df)
    y_test = test_df[config.LABEL_NAME].to_numpy()

    # 1. Logistic scorecard baseline vs LightGBM.
    lgbm = train_model(X_train, y_train)
    logistic = train_logistic(X_train, y_train)
    comparison = {
        "logistic": _scored(y_test, logistic.predict_proba(X_test)[:, 1]),
        "lightgbm": _scored(y_test, lgbm.predict_proba(X_test)[:, 1]),
    }

    # 2. Probability calibration
    fit_df, calib_df = time_based_split(train_df, test_frac=0.2)
    base = train_model(engineer_features(fit_df), fit_df[config.LABEL_NAME])
    p_calib = base.predict_proba(engineer_features(calib_df))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        p_calib, calib_df[config.LABEL_NAME].to_numpy()
    )
    p_test_base = base.predict_proba(X_test)[:, 1]
    calibration = {
        "brier_uncalibrated": brier_score(y_test, p_test_base),
        "brier_calibrated": brier_score(y_test, iso.transform(p_test_base)),
    }

    # 3. Temporal stability: expanding-window ROC-AUC per vintage year.
    folds = []
    for test_year, fold_train, fold_test in temporal_folds(df, min_train_years=3):
        y_ft = fold_test[config.LABEL_NAME].to_numpy()
        if len(fold_test) < 500 or len(np.unique(y_ft)) < 2:
            continue
        if len(fold_train) > FOLD_TRAIN_CAP:
            fold_train = fold_train.sample(FOLD_TRAIN_CAP, random_state=config.RANDOM_SEED)
        model = train_model(engineer_features(fold_train), fold_train[config.LABEL_NAME])
        p_ft = model.predict_proba(engineer_features(fold_test))[:, 1]
        folds.append(
            {
                "test_year": test_year,
                "n": int(len(fold_test)),
                "default_rate": float(y_ft.mean()),
                "roc_auc": float(classification_metrics(y_ft, p_ft)["roc_auc"]),
            }
        )

    result = {"comparison": comparison, "calibration": calibration, "temporal": folds}
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_PATH.write_text(json.dumps(result, indent=2))

    print("model comparison (test):")
    for name, m in comparison.items():
        print(f"  {name:<9} ROC-AUC {m['roc_auc']:.4f}  PR-AUC {m['pr_auc']:.4f}  Brier {m['brier']:.4f}")
    print(
        f"calibration Brier: {calibration['brier_uncalibrated']:.4f} "
        f"-> {calibration['brier_calibrated']:.4f}"
    )
    print("temporal ROC-AUC by vintage year:")
    for f in folds:
        print(f"  {f['test_year']}: {f['roc_auc']:.4f}  (n={f['n']:,}, default {f['default_rate']:.1%})")
    print(f"saved -> {VALIDATION_PATH}")


if __name__ == "__main__":
    main()
