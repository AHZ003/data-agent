"""DataSource — the engine-agnostic description of what the Coder can query.

A DataSource is a set of tables with columns, types, keys and a few
sample values, plus the SQL dialect the engine speaks. It is the single
thing the Coder prompt is rendered from, so a one-CSV upload, a
multi-CSV workspace and a Spider/BIRD SQLite file all look the same to
the LLM.

It is built by *introspecting the live connection* rather than the
source DataFrames, so types and sample values reflect what is actually
stored (e.g. pandas datetimes land in SQLite as 'YYYY-MM-DD HH:MM:SS'
text, and the prompt should show exactly that).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# Distinct counts need a full scan per column; skip them on big tables
# (BIRD has multi-million-row tables) and fall back to cheap samples.
_DISTINCT_COUNT_MAX_ROWS = 100_000
_SAMPLE_VALUE_MAX_CHARS = 40


@dataclass
class Column:
    name: str
    type: str
    nullable: bool = True
    sample_values: list = field(default_factory=list)
    distinct_count: Optional[int] = None
    description: Optional[str] = None


@dataclass
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass
class Table:
    name: str
    columns: list[Column]
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    row_count: Optional[int] = None

    def column(self, name: str) -> Optional[Column]:
        return next((c for c in self.columns if c.name.lower() == name.lower()), None)


@dataclass
class DataSource:
    name: str
    tables: list[Table]
    dialect: str = "sqlite"  # "sqlite" | "duckdb" | "bigquery"

    def table(self, name: str) -> Optional[Table]:
        return next((t for t in self.tables if t.name.lower() == name.lower()), None)

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    # ── Constructors ────────────────────────────────────────────────────

    @classmethod
    def from_sqlite_conn(
        cls, conn: sqlite3.Connection, name: str = "sqlite", n_samples: int = 5
    ) -> "DataSource":
        """Introspect every user table on an open SQLite connection."""
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ).fetchall()
        tables = [_introspect_sqlite_table(conn, r[0], n_samples) for r in rows]
        return cls(name=name, tables=tables, dialect="sqlite")

    @classmethod
    def from_sqlite(cls, path: str, name: Optional[str] = None, n_samples: int = 5) -> "DataSource":
        """Introspect a SQLite file, opened read-only."""
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return cls.from_sqlite_conn(conn, name=name or path, n_samples=n_samples)
        finally:
            conn.close()

    @classmethod
    def from_dataframes(cls, frames: dict[str, pd.DataFrame], name: str = "workspace") -> "DataSource":
        """Describe DataFrames as they would look once loaded into SQLite."""
        conn = sqlite3.connect(":memory:")
        try:
            for table_name, df in frames.items():
                df.to_sql(table_name, conn, index=False)
            return cls.from_sqlite_conn(conn, name=name)
        finally:
            conn.close()

    # ── Prompt rendering ────────────────────────────────────────────────

    def to_prompt(self, max_samples: int = 3) -> str:
        """Compact, LLM-friendly schema: one line per column.

        Example:
            Table "orders" (1,204 rows)
              "order_id" INTEGER PRIMARY KEY
              "customer_id" INTEGER -> "customers"."id"
              "region" TEXT -- 4 distinct; e.g. 'West', 'East', 'South'
        """
        q = (lambda s: f"`{s}`") if self.dialect == "bigquery" else (lambda s: f'"{s}"')
        blocks = []
        for t in self.tables:
            size = f" ({t.row_count:,} rows)" if t.row_count is not None else ""
            header = f"Table {q(t.name)}{size}"
            if len(t.primary_key) > 1:
                header += " PRIMARY KEY (" + ", ".join(q(c) for c in t.primary_key) + ")"
            lines = [header]
            fks = {fk.column.lower(): fk for fk in t.foreign_keys}
            # A composite key is shown once on the table line, not per column.
            pk = {c.lower() for c in t.primary_key} if len(t.primary_key) == 1 else set()
            for c in t.columns:
                line = f"  {q(c.name)} {c.type or 'ANY'}"
                if c.name.lower() in pk:
                    line += " PRIMARY KEY"
                fk = fks.get(c.name.lower())
                if fk:
                    line += f" -> {q(fk.ref_table)}.{q(fk.ref_column)}"
                notes = []
                if c.distinct_count is not None:
                    notes.append(f"{c.distinct_count:,} distinct")
                if c.sample_values:
                    notes.append("e.g. " + ", ".join(_fmt_value(v) for v in c.sample_values[:max_samples]))
                if c.description:
                    notes.append(c.description)
                if notes:
                    line += " -- " + "; ".join(notes)
                lines.append(line)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _introspect_sqlite_table(conn: sqlite3.Connection, table: str, n_samples: int) -> Table:
    qt = _quote(table)
    info = conn.execute(f"PRAGMA table_info({qt})").fetchall()
    # table_info rows: (cid, name, type, notnull, dflt_value, pk)
    pk = [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])]
    fks = [
        # foreign_key_list rows: (id, seq, table, from, to, ...). `to` is
        # NULL when the FK targets the parent's implicit primary key.
        ForeignKey(column=r[3], ref_table=r[2], ref_column=r[4] or "")
        for r in conn.execute(f"PRAGMA foreign_key_list({qt})").fetchall()
    ]
    row_count = conn.execute(f"SELECT COUNT(*) FROM {qt}").fetchone()[0]

    columns = []
    for _, col, ctype, notnull, _, _ in info:
        qc = _quote(col)
        if row_count <= _DISTINCT_COUNT_MAX_ROWS:
            distinct = conn.execute(f"SELECT COUNT(DISTINCT {qc}) FROM {qt}").fetchone()[0]
            # Most frequent values are the most useful examples for filters.
            samples = conn.execute(
                f"SELECT {qc} FROM {qt} WHERE {qc} IS NOT NULL "
                f"GROUP BY {qc} ORDER BY COUNT(*) DESC, {qc} LIMIT ?",
                (n_samples,),
            ).fetchall()
        else:
            distinct = None
            samples = conn.execute(
                f"SELECT DISTINCT {qc} FROM {qt} WHERE {qc} IS NOT NULL LIMIT ?",
                (n_samples,),
            ).fetchall()
        columns.append(
            Column(
                name=col,
                type=ctype or "",
                nullable=not notnull,
                sample_values=[s[0] for s in samples],
                distinct_count=distinct,
            )
        )
    return Table(name=table, columns=columns, primary_key=pk, foreign_keys=fks, row_count=row_count)


def _fmt_value(v) -> str:
    if isinstance(v, str):
        s = v if len(v) <= _SAMPLE_VALUE_MAX_CHARS else v[: _SAMPLE_VALUE_MAX_CHARS - 1] + "…"
        return "'" + s.replace("'", "''") + "'"
    if isinstance(v, bytes):
        return "<blob>"
    return str(v)
