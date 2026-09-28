# Data at scale

DataAgent runs the same guarded text-to-SQL path on three engines behind one
interface (`core/engine.py`).

| Engine | For | Safety layers beyond the shared AST guard |
|---|---|---|
| SQLite (`core/database.py`) | Uploads, Spider/BIRD files | sqlite3 authorizer (read-only), progress-handler timeout |
| DuckDB (`core/duckdb_engine.py`) | Large CSV/Parquet uploads | external file access disabled and config locked after load; read-only files; interrupt timer |
| BigQuery (`core/bigquery_engine.py`) | Warehouse tables | free dry run before every query, refuse/confirm thresholds, `maximum_bytes_billed`, job timeout |

The guard (`core/database._guard`) parses in the engine's dialect and, for
every dialect, rejects file-reading functions (`read_csv`, `read_parquet`,
`glob`, …) and path-like table names, because DuckDB will otherwise read any
file the process can (`SELECT * FROM '/etc/passwd'`). Dialect-specific prompt
rules live in `prompts/dialect_{sqlite,duckdb,bigquery}.md`.

## SQLite vs DuckDB

`python -m benchmarks.engine_bench --rows 1000000 10000000` (Apple Silicon
laptop, median of 5, through each engine's guarded `execute_query`):

| Query | 1M rows: SQLite → DuckDB | 10M rows: SQLite → DuckDB |
|---|---|---|
| group-by sum | 203 → 3 ms (66×) | 2,169 → 16 ms (137×) |
| filtered count | 28 → 2 ms (15×) | 290 → 10 ms (30×) |
| monthly trend | 278 → 5 ms (61×) | 3,003 → 25 ms (121×) |
| top-10 categories | 234 → 3 ms (74×) | 2,553 → 15 ms (165×) |

Load time for 10M rows: SQLite 6.2 s, DuckDB 2.8 s. DuckDB is columnar and
vectorized and uses all cores, SQLite is row-oriented and single-threaded, so
part of the gap is parallelism. At 10M rows the difference is between an
interactive answer and a multi-second wait on every question.

## BigQuery cost guard

Every generated query is dry-run first (free) and priced at on-demand rates
(`BQ_USD_PER_TIB`, default $6.25/TiB):

- above `BQ_MAX_BYTES` (default 200 GiB): refused. The error says how much the
  query would scan and asks for a partition filter, and it goes back to the
  Coder, so self-repair usually produces a cheaper query (tested in
  `tests/test_bigquery_engine.py`);
- above `BQ_CONFIRM_BYTES` (default 10 GiB): returns `CostConfirmationRequired`,
  which is terminal for the Coder; the UI/API asks the user and re-runs with
  `confirm_cost=True`;
- the real job carries `maximum_bytes_billed`, so BigQuery fails the job
  rather than bill beyond the cap even if the estimate was wrong.

Schemas cost nothing to describe: column metadata comes from the table, sample
values from `list_rows` (free), and the partition and clustering columns are
flagged in the schema the Coder sees.

## Pending (needs GCP credentials)

- `python -m benchmarks.bigquery_eval --check`: validate the 12 scaffold gold
  queries with free dry runs, then extend `benchmarks/cases_bigquery.yaml` to
  30–50 cases.
- Bytes scanned with vs without partition hints
  (`bigquery_eval` vs `bigquery_eval --no-partition-hints`) and how often the
  cost guard fires.
