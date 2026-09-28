---
name: dialect_bigquery
version: 1
description: BigQuery rules for the coder, incl. cost-aware scanning.
---
BigQuery-specific notes (queries are billed by bytes scanned):
- Select only the columns you need; never SELECT *
- ALWAYS filter on the partition column when the schema marks one (see the table notes), e.g.
  WHERE DATE(pickup_datetime) BETWEEN '2022-01-01' AND '2022-01-31'
- Quote identifiers with backticks, never double quotes (double quotes are strings in BigQuery)
- Use fully qualified table names exactly as shown in the schema, in backticks: `project.dataset.table`
- Dates: DATE_TRUNC(d, MONTH), EXTRACT(YEAR FROM d), FORMAT_DATE('%Y-%m', d), TIMESTAMP_DIFF(a, b, MINUTE)
- Use SAFE_DIVIDE(a, b) for ratios; APPROX_QUANTILES(x, 100)[OFFSET(50)] for a median
