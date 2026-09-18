{#
    Range contract for a numeric column.

    dbt ships not_null, unique, accepted_values and relationships, but no range
    test. Rather than pull in dbt_utils for this one macro (nothing else in the
    project needs it).

    NULLs pass: whether a column may be null is the job of the not_null test, so
    keeping the two concerns separate means a failure names one problem, not two.
#}
{% test accepted_range(model, column_name, min_value=none, max_value=none) %}

select
    {{ column_name }} as value
from {{ model }}
where {{ column_name }} is not null
  and (
      false
      {% if min_value is not none %} or {{ column_name }} < {{ min_value }} {% endif %}
      {% if max_value is not none %} or {{ column_name }} > {{ max_value }} {% endif %}
  )

{% endtest %}
