"""Data loading for the credit-risk pipeline.

Reads the raw Lending Club file and turns it into a labeled frame by applying the target definition from config.
Kept separate from feature engineering so the notebook, the tests, and the training code all load data the same way.

Design:
    - load_raw is the ONLY place that touches the raw file. Everything else works on the returned DataFrame,
      so the on-disk format is an implementation detail.
    - add_label is pure with respect to the input frame (returns a copy) and encodes the single business rule that defines the problem:
      which loans are defaults, and which are dropped because their outcome is unknown.
    - load_labeled reads only the allowlisted source columns plus the target, so the full-dataset load stays light on memory
      (about two dozen columns rather than the ~150 in the raw file).
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from credit_risk import config


def load_raw(nrows: int | None = None, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load the raw accepted-loans file.

    Pass `nrows` to read only the first N rows for fast iteration during EDA.
    Pass `columns` to read only a subset of columns (much lighter on memory).
    Leave both None to load the full file.
    """
    if not config.RAW_ACCEPTED_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found at {config.RAW_ACCEPTED_FILE}. "
            "See data/README.md to download it from Kaggle."
        )
    return pd.read_csv(
        config.RAW_ACCEPTED_FILE,
        compression="gzip",
        low_memory=False,
        nrows=nrows,
        usecols=list(columns) if columns is not None else None,
    )


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the binary `default` label and drop in-progress loans.

    Rows whose loan_status has no known outcome (Current, Late, ...) are removed, because we cannot train on an outcome that has not happened yet.
    Returns a new frame; the input is left untouched.
    """
    labels = df[config.TARGET_COLUMN].map(config.resolve_label)
    keep = labels.notna()
    labeled = df.loc[keep].copy()
    labeled[config.LABEL_NAME] = labels.loc[keep].astype(int)
    return labeled


def load_labeled(nrows: int | None = None) -> pd.DataFrame:
    """Load only the columns needed for modeling and attach the label.

    Reads the allowlisted feature-source columns plus the target column (not the full ~150),
    then applies add_label. This is the entry point training uses.
    """
    columns = sorted(config.FEATURE_SOURCE_COLUMNS | {config.TARGET_COLUMN})
    return add_label(load_raw(nrows=nrows, columns=columns))


def load_for_calibration(nrows: int | None = None) -> pd.DataFrame:
    """Load the feature-source columns, the realized-economics columns, and the target.
    Used only by the threshold study for cost calibration; the economics columns are never fed to the model.
    """
    columns = sorted(
        config.FEATURE_SOURCE_COLUMNS
        | set(config.ECONOMICS_COLUMNS)
        | {config.TARGET_COLUMN}
    )
    return add_label(load_raw(nrows=nrows, columns=columns))
