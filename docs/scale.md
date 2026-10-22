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

## NYC taxi on BigQuery: results

**Data.** The public `tlc_yellow_trips_2022` table is not partitioned or
clustered, so every query scans each column it references in full.
`scripts/bigquery_setup.py` copies it into the project the way a data team
would store a 36M-row fact table: `nyc_taxi.trips_2022` partitioned by
`DATE(pickup_datetime)` and clustered by `pickup_location_id` (36,255,983
rows, 6.4 GiB), plus `nyc_taxi.zones` (260 zones). The source has no December
data (57 stray rows on Dec 1); the table description says so, and the Coder
sees it. One-time cost about $0.04.

**Golden set.** 40 questions (`benchmarks/cases_bigquery.yaml`): 31 name a
date range, 9 cover the whole year, 13 need a join to `zones`. Every gold query
was run and its answer checked by hand. Gemini 3.8 Flash, one run each:

| | Partition hints on | Hints removed |
|---|---|---|
| EX, strict (Spider rule) | 75.0% [60.0, 87.5] | 72.5% [57.5, 85.0] |
| EX, extra columns allowed | **97.5%** [92.5, 100] | 95.0% [87.5, 100] |
| Mean scanned, date-range questions | 0.060 GiB | 0.060 GiB |
| Mean scanned, all questions | 0.23 GiB | 0.20 GiB |
| Cost guard / confirmation fired (10 GiB default) | 0 | 0 |

- **Strict vs extra columns.** 9 of the 10 strict misses return the right rows
  plus the measure they ranked by ("which hour had the most pickups" → hour and
  count). The coder prompt asks for that on purpose: in the live demo a bare
  "Central" with no number read as a bug. Strict matching scores it wrong, so
  both numbers are reported.
- **The one real miss** (bq_040) filters on `'Times Square/Theatre District'`;
  the stored value is `'Times Sq/Theatre District'`. The value index that fixes
  this on SQLite/DuckDB isn't wired to BigQuery yet.
- **Partition hints don't matter here** (1 question differs, McNemar p = 1.0;
  same bytes on date-range questions). The questions name their dates, so the
  model filters on `pickup_datetime` with or without the schema note.

**What the layout saves.** The same 40 generated queries, dry-run against the
unpartitioned public table (`outputs/bigquery/layout_comparison.json`):

| Questions | Partitioned + clustered | Unpartitioned | Saved |
|---|---|---|---|
| Date range (31) | 0.060 GiB | 0.607 GiB | **90%** |
| Whole year (9) | 0.813 GiB | 0.836 GiB | 3% |
| All (40) | 0.229 GiB | 0.658 GiB | 65% |

The bytes saving comes from the table design plus the model's habit of
filtering on dates. Prompt hints add nothing when the question already names a
range. At on-demand prices the 40 questions cost $0.056 on the partitioned
table and $0.161 on the public one.

**In the demo.** The "NYC taxi trips" workspace shows the dry-run size and
price under every answer. With the demo's 1 GiB confirmation threshold, 2 of the
40 golden questions (the whole-year JFK tip percentage, 1.6 GiB, and the
whole-year long-trip fare, 1.35 GiB) stop for a "Run it" click; 8 GiB is the
hard cap, above the whole table's 6.4 GiB.

Reproduce:

```bash
python scripts/bigquery_setup.py --project $BQ_PROJECT
python -m benchmarks.bigquery_eval --check            # free dry runs
python -m benchmarks.bigquery_eval                    # hints on  (~$0.25 Gemini + <$0.06 BigQuery)
python -m benchmarks.bigquery_eval --no-partition-hints
```
