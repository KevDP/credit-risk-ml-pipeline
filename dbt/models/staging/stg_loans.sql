-- Raw Lending Club accepted loans, restricted to the application-time source
-- columns plus the outcome field, with the business label resolved.
--
-- Mirrors credit_risk.data.load_labeled: the same column allowlist and the same
-- target definition, so the SQL path and the Python path start from identical
-- rows. Loans whose outcome is not yet known are dropped rather than labeled,
-- because we cannot supervise on an outcome that has not happened. Here that
-- drop is expressed as an inner join against the label seed instead of a
-- hardcoded NOT IN list, so the set of known outcomes has exactly one
-- definition (config.py, via the seed).
--
-- Why all_varchar and explicit casts instead of type sniffing:
--
--   The export carries Lending Club summary rows *inside* the data, not only at
--   the end. Line 421,097 reads "Total amount funded in policy code 1:
--   6417608175" followed by empty fields. Sniffing types from the first 20,480
--   rows infers id as BIGINT and the build then dies on that row.
--
--   Reading every column as text and casting explicitly removes the guess. A
--   value that cannot be parsed becomes NULL and is caught by the not_null
--   contracts as a named test failure with a row count, instead of an opaque
--   parser error. The pandas path never surfaced these rows at all: read_csv
--   falls back to object dtype and the summary rows were dropped by accident,
--   because their empty loan_status maps to no label.

with source as (

    select *
    from read_csv_auto(
        '{{ var("raw_accepted_file") }}',
        header = true,
        compression = 'gzip',
        all_varchar = true
    )

),

typed as (

    select
        -- Primary key. Kept as text: it is an identifier, not a quantity.
        id,

        -- Numeric features. Cast to DOUBLE rather than an integer type even
        -- where the domain is a count, because the pandas path carries these as
        -- float64 (NaN forces it) and the contract test compares values.
        try_cast(loan_amnt as double)            as loan_amnt,
        try_cast(annual_inc as double)           as annual_inc,
        try_cast(dti as double)                  as dti,
        try_cast(open_acc as double)             as open_acc,
        try_cast(pub_rec as double)              as pub_rec,
        try_cast(revol_bal as double)            as revol_bal,
        try_cast(revol_util as double)           as revol_util,
        try_cast(total_acc as double)            as total_acc,
        try_cast(delinq_2yrs as double)          as delinq_2yrs,
        try_cast(inq_last_6mths as double)       as inq_last_6mths,
        try_cast(mort_acc as double)             as mort_acc,
        try_cast(pub_rec_bankruptcies as double) as pub_rec_bankruptcies,

        -- Categorical features kept as text.
        home_ownership,
        verification_status,
        purpose,
        addr_state,

        -- Consumed only to derive features; never reach the mart raw.
        term,
        emp_length,
        try_cast(fico_range_low as double)  as fico_range_low,
        try_cast(fico_range_high as double) as fico_range_high,
        issue_d,
        earliest_cr_line,

        -- Outcome field. Post-origination, so it stops at staging.
        loan_status

    from source
    -- Drop the embedded summary rows on purpose. A real loan always has a
    -- numeric identifier, so this is the predicate that separates data from the
    -- export's own totals. Stated explicitly rather than left to the label join,
    -- which would discard them for the wrong reason.
    where try_cast(id as bigint) is not null

),

labeled as (

    select
        typed.*,
        labels.label as "default"
    from typed
    inner join {{ ref('loan_status_labels') }} as labels
        on typed.loan_status = labels.loan_status

)

select * from labeled
