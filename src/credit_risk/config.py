"""Central configuration for the credit-risk pipeline.

Single source of truth for filesystem paths, reproducibility settings, and the omain decisions that define the modeling problem.
Every other module imports from here so paths and the target definition never drift between the notebook, the training code, and the serving layer.

Design:

    - Paths resolve relative to the repository root, not the caller's cwd, so the
      same code works from a notebook, a script, or a Lambda package.

    - The target definition (which loan outcomes count as a default) lives here
      because it is a business decision, not an implementation detail. Changing
      it must change training, evaluation, and serving at once.
      
    - KNOWN_LEAKAGE_COLUMNS lists fields that are only known after a loan is
      originated. Feeding them to the model leaks the future. The list is a
      starting point, refined during EDA (Phase 1).
"""
from __future__ import annotations

from pathlib import Path

# Repository root: three levels up from this file (src/credit_risk/config.py).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DOCS_DIR = REPO_ROOT / "docs"

# Raw Lending Club file as published on Kaggle (wordsforthewise/lending-club).
# Not committed; see data/README.md for download instructions.
RAW_ACCEPTED_FILE = DATA_DIR / "accepted_2007_to_2018Q4.csv.gz"

# Reproducibility: a single seed shared by every stochastic step.
RANDOM_SEED = 42

# --- Target definition (business decision)
# The label is binary: 1 = the borrower failed to repay, 0 = the loan was repaid.
# Loans whose outcome is not yet known are dropped, not labeled, because we cannot supervise on an outcome that has not happened.
TARGET_COLUMN = "loan_status"
LABEL_NAME = "default"

# loan_status values that resolve to a bad outcome (default = 1).
DEFAULT_STATUSES = frozenset({
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
})

# loan_status values that resolve to a good outcome (default = 0).
PAID_STATUSES = frozenset({
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
})

# --- Leakage guard -----------------------------------------------------------
# Columns populated only AFTER origination. They correlate almost perfectly with
# the label because they describe what happened to the loan, not what was known
# at application time. Refined during EDA (see notebooks/).
KNOWN_LEAKAGE_COLUMNS = frozenset({
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "debt_settlement_flag",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    # Hardship / payment-plan fields: populated only if the loan later enters a
    # hardship plan, i.e. strictly post-origination. Surfaced during EDA as a
    # large block of near-100%-missing columns.
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "payment_plan_start_date",
    "orig_projected_additional_accrued_interest",
})

# --- Excluded by design: Lending Club's own risk-model outputs ---------------
# grade, sub_grade, and int_rate are LC's internal risk pricing. installment is a
# deterministic function of int_rate (given loan_amnt and term), so it smuggles
# the same signal back in. All are excluded from the primary model so it learns
# an INDEPENDENT risk view from borrower and bureau features. They return only in
# the ablation study.
LC_RISK_MODEL_COLUMNS = frozenset({
    "grade",
    "sub_grade",
    "int_rate",
    "installment",
})

# --- Feature allowlist (known at application time) ---------------------------
# An explicit allowlist beats a blocklist for leakage safety: only columns known
# when the loan is approved are eligible, so a post-origination field can never
# slip into the model.

# Numeric columns used directly as features.
RAW_NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "delinq_2yrs",
    "inq_last_6mths",
    "mort_acc",
    "pub_rec_bankruptcies",
]

# Categorical columns kept as-is (LightGBM handles the category dtype natively).
CATEGORICAL_FEATURES = [
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
]

# Raw columns parsed into numeric features by features.py.
TERM_COLUMN = "term"              # " 36 months" -> 36
EMP_LENGTH_COLUMN = "emp_length"  # "< 1 year".."10+ years" -> 0..10

# Raw columns consumed only to derive features (not used directly).
FICO_LOW_COLUMN = "fico_range_low"
FICO_HIGH_COLUMN = "fico_range_high"
ISSUE_DATE_COLUMN = "issue_d"
EARLIEST_CREDIT_LINE_COLUMN = "earliest_cr_line"

# The ordered list of engineered feature names the model actually sees.
FEATURE_COLUMNS = RAW_NUMERIC_FEATURES + [
    "term_months",
    "emp_length_years",
    "fico_score",
    "credit_history_length",
] + CATEGORICAL_FEATURES

# Raw source columns pulled from the dataset to build FEATURE_COLUMNS. The
# leakage-guard test asserts none of them is a leakage or risk-model column.
FEATURE_SOURCE_COLUMNS = frozenset(
    RAW_NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + [
        TERM_COLUMN,
        EMP_LENGTH_COLUMN,
        FICO_LOW_COLUMN,
        FICO_HIGH_COLUMN,
        ISSUE_DATE_COLUMN,
        EARLIEST_CREDIT_LINE_COLUMN,
    ]
)

# --- Realized economics (for cost calibration, NEVER model features) ---------
# Post-origination cashflow columns. They must never be model inputs (they live
# in KNOWN_LEAKAGE_COLUMNS), but they are the ground truth for estimating
# LGD/margin and for scoring a decision policy in dollars.
TOTAL_PAYMENT_COLUMN = "total_pymnt"
RECOVERIES_COLUMN = "recoveries"
LOAN_AMOUNT_COLUMN = "loan_amnt"
ECONOMICS_COLUMNS = (TOTAL_PAYMENT_COLUMN, RECOVERIES_COLUMN, LOAN_AMOUNT_COLUMN)


def resolve_label(loan_status: str) -> int | None:
    """Map a raw loan_status value to a binary label.

    Returns 1 for a default, 0 for a repaid loan, and None for loans whose outcome is still unknown 
    (which must be dropped from the training set rather than labeled).
    """
    if loan_status in DEFAULT_STATUSES:
        return 1
    if loan_status in PAID_STATUSES:
        return 0
    return None
