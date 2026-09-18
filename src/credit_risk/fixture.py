"""Build the CI regression fixture from the raw Lending Club export.

CI cannot use the real source because filesize and fetching it on every push would cost
minutes to a question an already answered question (the export is frozen at 2018Q4). 
So the contracts run against a small file committed to the repo.

This is a *regression* fixture, it deliberately carries the rows that broke the build once, so CI keeps proving they stay handled:

    - summary rows embedded mid-file ("Total amount funded in policy code 1: ...") which the staging filter must drop
    - 2007 launch-cohort rows with null annual_inc and earliest_cr_line, which the cohort-scoped not_null contracts must tolerate
    - in-progress loans, which the label join must drop
    - at least two rows for every emp_length, term, home_ownership and verification_status value, so the parsing CASE is fully exercised

Only the columns the staging model reads are written: a fixture missing one of them fails the model, which is the behaviour we want anyway.

Run with (needs the real file in data/):

    python -m credit_risk.fixture
"""
from __future__ import annotations

from pathlib import Path

from credit_risk import config

FIXTURE_PATH = config.REPO_ROOT / "tests" / "fixtures" / "lending_club_sample.csv"

SOURCE_COLUMNS: tuple[str, ...] = (
    config.LOAN_ID_COLUMN,
    *config.RAW_NUMERIC_FEATURES,
    *config.CATEGORICAL_FEATURES,
    config.TERM_COLUMN,
    config.EMP_LENGTH_COLUMN,
    config.FICO_LOW_COLUMN,
    config.FICO_HIGH_COLUMN,
    config.ISSUE_DATE_COLUMN,
    config.EARLIEST_CREDIT_LINE_COLUMN,
    config.TARGET_COLUMN,
)

# Columns whose every distinct value must appear, so the SQL that parses them is never exercised by only a subset of its branches.
COVERAGE_COLUMNS: tuple[str, ...] = (
    config.EMP_LENGTH_COLUMN,
    config.TERM_COLUMN,
    "home_ownership",
    "verification_status",
)

ROWS_PER_COVERAGE_VALUE = 2
BASE_SAMPLE_SIZE = 2_000
SUMMARY_ROWS = 2
IN_PROGRESS_ROWS = 20


def _known_statuses_sql() -> str:
    """Render the resolved loan_status values as a SQL IN list."""
    statuses = sorted(config.DEFAULT_STATUSES | config.PAID_STATUSES)
    return ", ".join("'" + status.replace("'", "''") + "'" for status in statuses)


def build_query(source: str) -> str:
    """Return the SQL that assembles the fixture from `source`.

    Args:
        source: Path to the raw export, as a SQL string literal value.

    Returns A SELECT returning the fixture rows in a stable order.
    """
    columns = ", ".join(SOURCE_COLUMNS)
    known = _known_statuses_sql()

    coverage_blocks = "\n        union all\n".join(
        f"""
        (
            select {columns} from labeled
            qualify row_number() over (
                partition by {column} order by hash(id)
            ) <= {ROWS_PER_COVERAGE_VALUE}
        )"""
        for column in COVERAGE_COLUMNS
    )

    return f"""
    with raw as (
        select * from read_csv_auto('{source}', header = true, all_varchar = true)
    ),

    -- Real loans: a numeric id is what separates data from the export's totals.
    loans as (
        select * from raw where try_cast(id as bigint) is not null
    ),

    labeled as (
        select * from loans where {config.TARGET_COLUMN} in ({known})
    ),

    -- The rows that broke the build: export totals written mid-file.
    summary_rows as (
        select {columns} from raw
        where try_cast(id as bigint) is null
        order by id
        limit {SUMMARY_ROWS}
    ),

    -- The 2007 launch cohort, where the source stops reporting some fields.
    launch_cohort_nulls as (
        select {columns} from labeled
        where annual_inc is null or earliest_cr_line is null
    ),

    -- Loans whose outcome is not known yet: the label join must drop these.
    in_progress as (
        select {columns} from loans
        where {config.TARGET_COLUMN} not in ({known})
        order by hash(id)
        limit {IN_PROGRESS_ROWS}
    ),

    -- Two rows per distinct value of every column the SQL parses by cases.
    coverage as ({coverage_blocks}
    ),

    body as (
        select {columns} from labeled
        order by hash(id)
        limit {BASE_SAMPLE_SIZE}
    ),

    combined as (
        select * from summary_rows
        union all select * from launch_cohort_nulls
        union all select * from in_progress
        union all select * from coverage
        union all select * from body
    )

    select * from combined
    qualify row_number() over (partition by id order by 1) = 1
    order by id
    """


def write_fixture() -> Path:
    """Regenerate the fixture file from the raw export.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If the raw export is not present.
    """
    import duckdb

    if not config.RAW_ACCEPTED_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found at {config.RAW_ACCEPTED_FILE}. "
            "See data/README.md to download it from Kaggle."
        )

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    source = config.RAW_ACCEPTED_FILE.as_posix()

    connection = duckdb.connect()
    connection.execute(
        f"copy ({build_query(source)}) to '{FIXTURE_PATH.as_posix()}' "
        "(format csv, header true)"
    )
    connection.close()
    return FIXTURE_PATH


def main() -> None:
    path = write_fixture()
    rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
    size_kb = path.stat().st_size / 1024
    print(f"wrote {path.relative_to(config.REPO_ROOT)}: {rows} rows, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
