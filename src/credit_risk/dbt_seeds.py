"""Generate the dbt seeds from config, so SQL and Python cannot disagree silently.

Two contracts live in `config.py` and are also needed inside the warehouse:

    - the application-time allowlist (which columns may reach the feature mart)
    - the target definition (which loan_status values resolve to which label)

Duplicating either one by hand in SQL is how training/serving skew starts. Instead
the seeds are *derived* from config here, committed to the repo, and guarded from
both sides:

    - `dbt test` fails if the mart grows a column the allowlist seed does not list
    - `pytest` fails if a seed on disk no longer matches what config produces

So adding a column to the mart without declaring it breaks the build on purpose,
and editing config without regenerating breaks the test suite. Neither direction
can drift quietly.

Run with:

    python -m credit_risk.dbt_seeds
"""
from __future__ import annotations

import csv
from pathlib import Path

from credit_risk import config

# Seeds live inside the dbt project so `dbt seed` picks them up.
SEEDS_DIR = config.REPO_ROOT / "dbt" / "seeds"

ALLOWLIST_SEED = SEEDS_DIR / "application_time_allowlist.csv"
LABEL_SEED = SEEDS_DIR / "loan_status_labels.csv"

# Columns the mart carries that are not model features. They are declared
# explicitly rather than excluded by the test, so the allowlist stays a complete
# description of the mart instead of a partial one with hidden exceptions.
NON_FEATURE_MART_COLUMNS: tuple[tuple[str, str], ...] = (
    # Primary key. Lending Club's own loan identifier, known at application time.
    (config.LOAN_ID_COLUMN, "key"),
    # The supervised target. Derived from loan_status, which is post-outcome, so
    # it must never be read back as a feature.
    (config.LABEL_NAME, "label"),
    # Origination month. Known at application time and used as the out-of-time
    # split key and the partition grain (see split.time_based_split).
    (config.ISSUE_DATE_COLUMN, "partition"),
)


def build_allowlist_rows() -> list[tuple[str, str]]:
    """Return (column_name, role) rows describing every column the mart may hold.

    Returns:
        Feature columns in their canonical `config.FEATURE_COLUMNS` order, then
        the declared non-feature columns.
    """
    rows: list[tuple[str, str]] = [(name, "feature") for name in config.FEATURE_COLUMNS]
    rows.extend(NON_FEATURE_MART_COLUMNS)
    return rows


def build_label_rows() -> list[tuple[str, int]]:
    """Return (loan_status, label) rows for every status with a known outcome.

    Statuses absent from this seed are in-progress loans: the staging model drops
    them, matching `data.add_label`.
    """
    rows = [(status, 1) for status in sorted(config.DEFAULT_STATUSES)]
    rows.extend((status, 0) for status in sorted(config.PAID_STATUSES))
    return rows


def _write_csv(path: Path, header: tuple[str, ...], rows: list) -> None:
    """Write `rows` as a CSV with LF endings so the file is identical on any OS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_seeds() -> list[Path]:
    """Regenerate both seed files from config.

    Returns:
        The paths written, in a stable order.
    """
    _write_csv(ALLOWLIST_SEED, ("column_name", "role"), build_allowlist_rows())
    _write_csv(LABEL_SEED, ("loan_status", "label"), build_label_rows())
    return [ALLOWLIST_SEED, LABEL_SEED]


def main() -> None:
    for path in write_seeds():
        print(f"wrote {path.relative_to(config.REPO_ROOT)}")


if __name__ == "__main__":
    main()
