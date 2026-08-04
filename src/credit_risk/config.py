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
})


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
