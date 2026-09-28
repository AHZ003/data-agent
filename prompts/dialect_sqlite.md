---
name: dialect_sqlite
version: 1
description: SQLite rules for the coder (moved from coder_agent v4).
---
SQLite-specific notes (CRITICAL — violating these causes execution errors):
- Wrap table and column names in double quotes
- SQLite has NO CORR(), STDDEV(), VARIANCE(), PERCENTILE_CONT, or MEDIAN functions. NEVER use them.
- For correlation between columns x and y, compute manually:
  SELECT (SUM("x" * "y") - SUM("x") * SUM("y") / COUNT(*)) /
         (SQRT((SUM("x" * "x") - SUM("x") * SUM("x") / COUNT(*)) *
               (SUM("y" * "y") - SUM("y") * SUM("y") / COUNT(*)))) AS correlation
  FROM "table"
- For standard deviation: SQRT(AVG("x" * "x") - AVG("x") * AVG("x"))
- For median: use a subquery with LIMIT 1 OFFSET COUNT(*)/2 after ORDER BY
- Dates are stored as TEXT; use strftime()/date() for date parts and ranges
- Integer division truncates: multiply by 1.0 for ratios and percentages
- ALWAYS return at least one row — if computing a scalar (avg, sum, correlation), wrap in a SELECT that guarantees output
