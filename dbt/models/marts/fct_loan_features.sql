-- The model-ready feature mart: the SQL counterpart of credit_risk.features.engineer_features.
--
-- Only application-time columns appear here. The singular test
-- assert_mart_matches_allowlist.sql fails the build if this model grows a
-- column the allowlist seed does not declare, and tests/test_dbt_contract.py
-- fails if these transformations stop agreeing with the Python ones row by row.
-- That second test is what keeps the emp_length CASE below from drifting away
-- from _EMP_LENGTH_MAP without needing a third seed to hold eleven values.
--
-- issue_d is carried as a DATE: it is known at application time, it is the
-- out-of-time split key (see split.time_based_split) and it is the partition
-- grain for backfills.

with source as (

    select * from {{ ref('stg_loans') }}

),

parsed as (

    select
        -- Primary key.
        id,

        -- Passed through unchanged.
        loan_amnt,
        annual_inc,
        dti,
        open_acc,
        pub_rec,
        revol_bal,
        revol_util,
        total_acc,
        delinq_2yrs,
        inq_last_6mths,
        mort_acc,
        pub_rec_bankruptcies,

        -- ' 36 months' -> 36.0. Non-parseable values become NULL.
        cast(regexp_extract(term, '(\d+)') as double) as term_months,

        -- '< 1 year'..'10+ years' -> 0..10. Unknown stays NULL.
        case emp_length
            when '< 1 year'  then 0.0
            when '1 year'    then 1.0
            when '2 years'   then 2.0
            when '3 years'   then 3.0
            when '4 years'   then 4.0
            when '5 years'   then 5.0
            when '6 years'   then 6.0
            when '7 years'   then 7.0
            when '8 years'   then 8.0
            when '9 years'   then 9.0
            when '10+ years' then 10.0
        end as emp_length_years,

        -- Lending Club reports a FICO band; the model sees its midpoint.
        (fico_range_low + fico_range_high) / 2.0 as fico_score,

        -- Years between the oldest credit line and origination. Both dates look
        -- like 'Dec-2011' and parse to the first of the month, so the day count
        -- is exact and matches the pandas Timedelta path.
        date_diff(
            'day',
            strptime(earliest_cr_line, '%b-%Y'),
            strptime(issue_d, '%b-%Y')
        ) / 365.25 as credit_history_length,

        -- Categorical features.
        home_ownership,
        verification_status,
        purpose,
        addr_state,

        -- Label and partition key.
        "default",
        cast(strptime(issue_d, '%b-%Y') as date) as issue_d

    from source

)

select * from parsed
