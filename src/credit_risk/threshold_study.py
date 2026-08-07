"""Threshold study: does calibrating cost from data beat a fixed assumption?

Compares two ways of setting the approve/decline cutoff:
    - fixed illustrative costs (--fixed-lgd, --fixed-margin)
    - costs estimated from the training vintages' realized cashflows.

Both cutoffs are derived without touching the test set, then scored on the test set's REALIZED dollars.
An oracle cutoff (the post-hoc profit maximum on test) is reported as an unachievable upper bound, and "approve all" as a floor.
Results go to experiments/ (a JSON summary and the full profit curve as CSV).

Examples:
    python -m credit_risk.threshold_study
    python -m credit_risk.threshold_study --sample 300000
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from credit_risk import config
from credit_risk.calibrate import estimate_lgd_margin, profit_threshold, realized_net
from credit_risk.data import load_for_calibration
from credit_risk.evaluate import policy_outcome
from credit_risk.features import engineer_features
from credit_risk.split import time_based_split
from credit_risk.train import train_model

STUDY_PATH = config.EXPERIMENTS_DIR / "threshold_study.json"
CURVE_PATH = config.EXPERIMENTS_DIR / "profit_curve.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None,
                        help="Read only the first N rows (fast dev runs).")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--fixed-lgd", type=float, default=0.65,
                        help="Illustrative loss-given-default assumption.")
    parser.add_argument("--fixed-margin", type=float, default=0.12,
                        help="Illustrative profit-margin assumption.")
    args = parser.parse_args()

    df = load_for_calibration(nrows=args.sample)
    train_df, test_df = time_based_split(df, test_frac=args.test_frac)

    X_train = engineer_features(train_df)
    y_train = train_df[config.LABEL_NAME]
    X_test = engineer_features(test_df)
    y_test = test_df[config.LABEL_NAME].to_numpy()

    model = train_model(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    net_test = realized_net(test_df).to_numpy()

    # Cost models -> cutoffs. The empirical costs come from TRAIN only, so no
    # test information leaks into either cutoff.
    lgd_emp, margin_emp = estimate_lgd_margin(train_df, y_train)
    thr_fixed = profit_threshold(args.fixed_lgd, args.fixed_margin)
    thr_emp = profit_threshold(lgd_emp, margin_emp)

    # Full profit curve on test + the post-hoc oracle (an upper bound only).
    grid = np.linspace(0.0, 1.0, 101)
    curve = [policy_outcome(y_test, proba, net_test, t) for t in grid]
    oracle = max(curve, key=lambda row: row["realized_profit"])

    result = {
        "empirical_lgd": lgd_emp,
        "empirical_margin": margin_emp,
        "fixed_lgd": args.fixed_lgd,
        "fixed_margin": args.fixed_margin,
        "fixed": policy_outcome(y_test, proba, net_test, thr_fixed),
        "empirical": policy_outcome(y_test, proba, net_test, thr_emp),
        "approve_all": policy_outcome(y_test, proba, net_test, 1.0),
        "oracle": oracle,
    }

    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_PATH.write_text(json.dumps(result, indent=2))
    pd.DataFrame(curve).to_csv(CURVE_PATH, index=False)

    print(f"empirical LGD={lgd_emp:.3f}  margin={margin_emp:.3f}")
    for name in ("fixed", "empirical", "oracle", "approve_all"):
        row = result[name]
        print(
            f"{name:>11}: cutoff {row['threshold']:.3f} | "
            f"profit ${row['realized_profit']:,.0f} | "
            f"approval {row['approval_rate']:.1%} | "
            f"book default {row['book_default_rate']:.1%}"
        )
    print(f"saved -> {STUDY_PATH} and {CURVE_PATH}")


if __name__ == "__main__":
    main()
