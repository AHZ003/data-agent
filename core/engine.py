"""Engine interface: where generated SQL runs.

Implementations: SQLite (`core.database.Database`, uploads and benchmark
files), DuckDB (`core.duckdb_engine`, large CSV/Parquet), BigQuery
(`core.bigquery_engine`, warehouse tables with a dry-run cost guard).
The Coder, the retriever and the benchmarks only use this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from core.datasource import DataSource


@dataclass(frozen=True)
class CostEstimate:
    bytes_processed: int
    usd: float

    def fmt(self) -> str:
        gib = self.bytes_processed / 2**30
        return f"{gib:,.2f} GiB scanned (~${self.usd:,.4f})"


class Engine(ABC):
    dialect: str = "sqlite"
    name: str = "workspace"
    table_name: str = ""          # primary table, for single-table UI paths
    timeout_seconds: float = 10.0
    max_rows: int = 1000

    @abstractmethod
    def datasource(self) -> DataSource:
        """Describe every queryable table (cached)."""

    @abstractmethod
    def execute_query(self, sql: str, max_rows: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Guarded, read-only execution of untrusted SQL: (result, error).
        Must be safe to call from several threads at once."""

    @abstractmethod
    def fetch(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Trusted internal queries (introspection, value index). Never LLM SQL."""

    def dry_run(self, sql: str) -> Optional[CostEstimate]:
        """Estimated cost of `sql` before running it; None if the engine is free."""
        return None

    def close(self) -> None:
        pass
