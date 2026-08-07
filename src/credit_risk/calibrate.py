"""Cost calibration from realized cashflows.

Estimates loss-given-default (LGD) and profit margin as fractions of the loan amount directly from the historical outcomes in the data,
and computes the realized net dollars per loan. These use post-origination columns (total_pymnt, recoveries): 
valid for VALUING outcomes, never as model inputs.

The profit-maximizing decision cutoff follows analytically from the cost model: approve a loan when its expected profit is positive,
i.e. when P(default) < margin / (LGD + margin). Because both costs scale with the loan amount, the amount cancels and the cutoff is the same for every loan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk import config


def realized_net(df: pd.DataFrame) -> pd.Series:
    """Actual dollars each loan returned: total_pymnt + recoveries - loan_amnt.
    Positive is profit, negative is loss."""
    return (
        df[config.TOTAL_PAYMENT_COLUMN]
        + df[config.RECOVERIES_COLUMN]
        - df[config.LOAN_AMOUNT_COLUMN]
    )


def estimate_lgd_margin(df: pd.DataFrame, label: pd.Series) -> tuple[float, float]:
    """Estimate LGD and margin as fractions of loan_amnt from realized cashflows.

    LGD is the mean loss fraction over defaulted loans; margin is the mean profit fraction over repaid loans. Both are floored at 0.
    """
    net = realized_net(df).to_numpy()
    amount = df[config.LOAN_AMOUNT_COLUMN].to_numpy()
    label = np.asarray(label)
    is_default = label == 1
    is_paid = label == 0
    lgd = float(np.nanmean(np.clip(-net[is_default] / amount[is_default], 0, None)))
    margin = float(np.nanmean(np.clip(net[is_paid] / amount[is_paid], 0, None)))
    return lgd, margin


def profit_threshold(lgd: float, margin: float) -> float:
    """Probability cutoff that maximizes expected profit under the cost model: approve when P(default) < margin / (LGD + margin)."""
    return margin / (lgd + margin)
