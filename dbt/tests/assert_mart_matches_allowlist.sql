-- Leakage guard: the feature mart and the application-time allowlist must
-- describe exactly the same set of columns.
--
-- Fails in both directions on purpose:
--
--   undeclared_column: the mart grew a column nobody declared. This is the case
--     that matters for leakage. A post-origination field can only reach the
--     model by appearing here first, and it breaks the build instead.
--
--   missing_column: the allowlist declares a column the mart does not build.
--     Silent feature loss, which downstream shows up as a quietly worse model
--     rather than as an error.
--
-- The allowlist is not hand-maintained: it is generated from config.py by
-- `python -m credit_risk.dbt_seeds`, and tests/test_dbt_contract.py fails if the
-- committed seed no longer matches what config produces.

-- The table name and schema come from ref(). That is not cosmetic: 
-- reading information_schema gives dbt no dependency to infer, so
-- without a ref() call this test gets scheduled before the mart exists and
-- reports every declared column as missing. Naming it through ref() puts the
-- edge in the graph and the test runs after the model it checks.

{%- set mart = ref('fct_loan_features') %}

with mart_columns as (

    select lower(column_name) as column_name
    from information_schema.columns
    where lower(table_name) = lower('{{ mart.identifier }}')
      and lower(table_schema) = lower('{{ mart.schema }}')

),

allowlist as (

    select lower(column_name) as column_name
    from {{ ref('application_time_allowlist') }}

),

undeclared as (

    select
        mart_columns.column_name,
        'undeclared_column' as failure_reason
    from mart_columns
    left join allowlist using (column_name)
    where allowlist.column_name is null

),

missing as (

    select
        allowlist.column_name,
        'missing_column' as failure_reason
    from allowlist
    left join mart_columns using (column_name)
    where mart_columns.column_name is null

)

select * from undeclared
union all
select * from missing
