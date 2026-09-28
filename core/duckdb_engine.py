"""DuckDB engine: columnar execution for large CSV/Parquet uploads.

SQLite is row-oriented and single-threaded; an aggregate over 10M rows
takes seconds. DuckDB is columnar and vectorized (see
benchmarks/engine_bench.py for measured numbers on this machine).

Safety model, parallel to core/database.py:

  1. The same sqlglot AST guard (in the duckdb dialect), which also
     blocks file-reading table functions (read_csv, read_parquet, glob…)
     and string-literal table paths (FROM 'x.csv').
  2. Engine level: after the data is loaded, `enable_external_access`
     is switched off and the configuration is locked, so SQL cannot read
     or write files or re-enable access even if layer 1 were bypassed.
     File databases are opened read_only.
  3. Timeout: a timer thread calls `connection.interrupt()`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import pandas as pd

from config import MAX_QUERY_ROWS, QUERY_TIMEOUT_SECONDS
from core.database import SQLGuardError, _guard
from core.datasource import Column, DataSource, ForeignKey, Table
from core.engine import Engine

_DISTINCT_COUNT_MAX_ROWS = 5_000_000  # columnar distinct counts are cheap


class DuckDBEngine(Engine):
    dialect = "duckdb"

    def __init__(self, conn: Optional[duckdb.DuckDBPyConnection] = None, name: str = "workspace"):
        self.conn = conn or duckdb.connect(":memory:")
        self.name = name
        self.table_name = ""
        self.timeout_seconds = QUERY_TIMEOUT_SECONDS
        self.max_rows = MAX_QUERY_ROWS
        self._datasource: Optional[DataSource] = None
        self._locked = False
        self._lock = threading.Lock()

    # ── Loading (before lock) ───────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str, name: Optional[str] = None) -> "DuckDBEngine":
        """Open an existing .duckdb database read-only."""
        eng = cls(duckdb.connect(path, read_only=True), name=name or Path(path).stem)
        eng.lock()
        names = eng.datasource().table_names
        eng.table_name = names[0] if names else ""
        return eng

    def load_file(self, path: str, table_name: str) -> None:
        """Load a CSV or Parquet file as a table (must happen before lock())."""
        self._require_unlocked()
        reader = "read_parquet" if str(path).lower().endswith(".parquet") else "read_csv_auto"
        self.conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM {reader}(?)', [str(path)])
        self._loaded(table_name)

    def load_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        self._require_unlocked()
        self.conn.register("__df", df)
        self.conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM __df')
        self.conn.unregister("__df")
        self._loaded(table_name)

    def lock(self) -> None:
        """Turn off file access for the rest of this connection's life."""
        if not self._locked:
            self.conn.execute("SET enable_external_access = false")
            self.conn.execute("SET lock_configuration = true")
            self._locked = True

    def _loaded(self, table_name: str) -> None:
        self.table_name = self.table_name or table_name
        self._datasource = None

    def _require_unlocked(self) -> None:
        if self._locked:
            raise RuntimeError("engine is locked; load data before the first query")

    # ── Engine interface ────────────────────────────────────────────────

    def fetch(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self.conn.execute(sql, list(params)).fetchall()

    def datasource(self) -> DataSource:
        if self._datasource is None:
            self._datasource = _introspect(self, self.name)
        return self._datasource

    def execute_query(self, sql: str, max_rows: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        try:
            safe_sql = _guard(sql, self.dialect)
        except SQLGuardError as e:
            return None, f"SQLGuardError: {e}"
        self.lock()
        timed_out = threading.Event()

        def _interrupt():
            timed_out.set()
            self.conn.interrupt()

        timer = threading.Timer(self.timeout_seconds, _interrupt)
        with self._lock:
            timer.start()
            try:
                cur = self.conn.execute(safe_sql)
                columns = [d[0] for d in cur.description or []]
                rows = cur.fetchmany(max_rows or self.max_rows)
                return pd.DataFrame.from_records(rows, columns=columns), None
            except duckdb.Error as e:
                if timed_out.is_set():
                    return None, f"QueryTimeout: query exceeded {self.timeout_seconds}s and was interrupted"
                return None, str(e).splitlines()[0]
            finally:
                timer.cancel()

    def close(self) -> None:
        self.conn.close()


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _introspect(eng: DuckDBEngine, name: str, n_samples: int = 5) -> DataSource:
    tables = []
    table_rows = eng.fetch(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' ORDER BY table_name"
    )
    constraints = eng.fetch(
        "SELECT table_name, constraint_type, constraint_column_names, "
        "referenced_table, referenced_column_names FROM duckdb_constraints() "
        "WHERE constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')"
    )
    for (tname,) in table_rows:
        cols = eng.fetch(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position", (tname,)
        )
        row_count = eng.fetch(f"SELECT COUNT(*) FROM {_q(tname)}")[0][0]
        pk, fks = [], []
        for t, ctype, ccols, rtable, rcols in constraints:
            if t != tname:
                continue
            if ctype == "PRIMARY KEY":
                pk = list(ccols)
            else:
                fks += [ForeignKey(c, rtable, r) for c, r in zip(ccols, rcols or [""] * len(ccols))]
        columns = []
        for cname, ctype, nullable in cols:
            qc = _q(cname)
            distinct = None
            if row_count <= _DISTINCT_COUNT_MAX_ROWS:
                distinct = eng.fetch(f"SELECT COUNT(DISTINCT {qc}) FROM {_q(tname)}")[0][0]
            samples = eng.fetch(
                f"SELECT {qc} FROM {_q(tname)} WHERE {qc} IS NOT NULL "
                f"GROUP BY {qc} ORDER BY COUNT(*) DESC, {qc} LIMIT {int(n_samples)}"
            )
            columns.append(Column(cname, ctype, nullable == "YES",
                                  [s[0] if isinstance(s[0], (str, int, float)) else str(s[0]) for s in samples],
                                  distinct))
        tables.append(Table(tname, columns, pk, fks, row_count))
    return DataSource(name=name, tables=tables, dialect="duckdb")
