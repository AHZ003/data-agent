"""BigQuery engine with a dry-run cost guard.

On BigQuery a bad query is not slow, it is expensive: on-demand pricing
bills bytes scanned, and `SELECT *` over a multi-TB table costs real
money. Every generated query goes through:

  1. The sqlglot AST guard in the bigquery dialect (read-only, one
     statement, no external/ML functions).
  2. A **dry run** (free) that reports bytes the query would scan.
     - above BQ_MAX_BYTES: refused, and the error tells the Coder how
       much it would scan so self-repair can add a partition filter or
       drop columns;
     - above BQ_CONFIRM_BYTES: returns a `CostConfirmationRequired`
       error (terminal for the Coder) so the UI/API can ask the user and
       re-run with `confirm_cost=True`.
  3. `maximum_bytes_billed` on the real job: BigQuery itself fails the
     job rather than bill beyond the cap, even if the estimate was off.
  4. A job timeout.

Introspection costs nothing: schemas come from table metadata, sample
values from `list_rows` (tabledata.list is free), and partition /
clustering columns are flagged in the schema so the Coder filters on
them (prompts/dialect_bigquery.md).

Requires the optional extra: `uv sync --extra bigquery`.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd

from config import MAX_QUERY_ROWS
from core.database import SQLGuardError, _guard
from core.datasource import Column, DataSource, Table
from core.engine import CostEstimate, Engine

USD_PER_TIB = float(os.getenv("BQ_USD_PER_TIB", "6.25"))  # on-demand, US multi-region
COST_SENTINEL = "CostConfirmationRequired"


def estimate(bytes_processed: int) -> CostEstimate:
    return CostEstimate(bytes_processed, round(bytes_processed / 2**40 * USD_PER_TIB, 6))


class BigQueryEngine(Engine):
    dialect = "bigquery"

    def __init__(
        self,
        tables: list[str],
        client=None,
        name: str = "bigquery",
        confirm_bytes: Optional[int] = None,
        max_bytes: Optional[int] = None,
        timeout_seconds: float = 120.0,
    ):
        """`tables` are fully qualified ids, e.g. 'bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022'."""
        if client is None:
            from google.cloud import bigquery
            client = bigquery.Client()
        self.client = client
        self.tables = tables
        self.name = name
        self.table_name = tables[0] if tables else ""
        self.confirm_bytes = confirm_bytes if confirm_bytes is not None else int(os.getenv("BQ_CONFIRM_BYTES", str(10 * 2**30)))
        self.max_bytes = max_bytes if max_bytes is not None else int(os.getenv("BQ_MAX_BYTES", str(200 * 2**30)))
        self.timeout_seconds = timeout_seconds
        self.max_rows = MAX_QUERY_ROWS
        self._datasource: Optional[DataSource] = None

    def _job_config(self, **kw):
        from google.cloud import bigquery
        return bigquery.QueryJobConfig(**kw)

    def dry_run(self, sql: str) -> CostEstimate:
        job = self.client.query(sql, job_config=self._job_config(dry_run=True, use_query_cache=False))
        return estimate(int(job.total_bytes_processed or 0))

    def execute_query(
        self, sql: str, max_rows: Optional[int] = None, confirm_cost: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        try:
            safe_sql = _guard(sql, self.dialect)
        except SQLGuardError as e:
            return None, f"SQLGuardError: {e}"
        try:
            cost = self.dry_run(safe_sql)
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e).splitlines()[0]}"
        if cost.bytes_processed > self.max_bytes:
            return None, (f"CostGuard: query would scan {cost.fmt()}, above the "
                          f"{self.max_bytes / 2**30:,.0f} GiB limit. Filter on the partition "
                          "column and select only needed columns.")
        if cost.bytes_processed > self.confirm_bytes and not confirm_cost:
            return None, f"{COST_SENTINEL}: query would scan {cost.fmt()}"
        try:
            job = self.client.query(safe_sql, job_config=self._job_config(maximum_bytes_billed=self.max_bytes))
            rows = job.result(timeout=self.timeout_seconds, max_results=max_rows or self.max_rows)
            return rows.to_dataframe(), None
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e).splitlines()[0]}"

    def fetch(self, sql: str, params: tuple = ()) -> list[tuple]:
        raise NotImplementedError("BigQuery introspection uses table metadata, not queries")

    def datasource(self) -> DataSource:
        if self._datasource is None:
            self._datasource = DataSource(
                name=self.name, tables=[self._describe(t) for t in self.tables], dialect="bigquery",
            )
        return self._datasource

    def _describe(self, table_id: str) -> Table:
        meta = self.client.get_table(table_id)
        part = getattr(getattr(meta, "time_partitioning", None), "field", None)
        cluster = set(getattr(meta, "clustering_fields", None) or [])
        try:  # free: tabledata.list, not a query
            samples = [dict(r.items()) for r in self.client.list_rows(meta, max_results=5)]
        except Exception:
            samples = []
        columns = []
        for f in meta.schema:
            notes = [f.description] if getattr(f, "description", None) else []
            if f.name == part:
                notes.append("PARTITION COLUMN: always filter on it to limit bytes scanned")
            elif f.name in cluster:
                notes.append("clustering column: filters on it are cheaper")
            vals = [s.get(f.name) for s in samples if s.get(f.name) is not None]
            columns.append(Column(
                name=f.name, type=f.field_type, nullable=f.mode != "REQUIRED",
                sample_values=[v if isinstance(v, (str, int, float)) else str(v) for v in vals[:3]],
                description="; ".join(notes) or None,
            ))
        return Table(name=table_id, columns=columns, row_count=getattr(meta, "num_rows", None))
