---
name: dialect_duckdb
version: 1
description: DuckDB rules for the coder.
---
DuckDB-specific notes:
- Wrap table and column names in double quotes
- Statistical aggregates exist: corr(x, y), stddev_samp(x), median(x), quantile_cont(x, 0.9)
- Dates: date_trunc('month', d), extract(year FROM d), d - INTERVAL 7 DAY; strftime(d, '%Y-%m')
- Integer division with / returns a double; use // for integer division
- Query only the tables in the schema; never read files (read_csv, read_parquet, paths)
