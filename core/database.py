"""SQLite database connection and guarded query execution.

Safety model
------------
The Coder agent emits SQL that runs against user data. The LLM is not
trusted to stay read-only on its own — a prompt-injected CSV or a
creative rephrasing can steer it toward destructive statements. Every
generated query passes three independent layers:

  1. **AST validation (sqlglot).** The query must parse, in the
     engine's dialect, to exactly one statement whose root is a SELECT
     or a set operation (UNION / INTERSECT / EXCEPT) of SELECTs; CTEs
     hang off the SELECT. Any DML/DDL/PRAGMA/ATTACH/transaction node
     *anywhere* in the tree is rejected. Because this works on the
     parse tree, not on text, string literals and quoted identifiers
     can't cause false positives (`REPLACE(...)`, a column named
     "update", `'it''s; fine'`) and comments can't hide anything.
  2. **Engine-level read-only (sqlite3 authorizer).** While a guarded
     query runs, SQLite itself denies every action except reads,
     SELECT, function calls and recursive CTEs. If layer 1 were ever
     bypassed, writes, ATTACH and PRAGMA still fail inside the engine.
  3. **Timeout (sqlite3 progress handler).** Long-running queries —
     e.g. a runaway recursive CTE — are interrupted after
     QUERY_TIMEOUT_SECONDS.

Results are capped at MAX_QUERY_ROWS via fetchmany, so a huge result
never materializes in memory. Tests live in tests/test_sql_guard.py.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Optional, Tuple

import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError
from sqlglot.tokens import TokenType

from config import DEFAULT_TABLE_NAME, MAX_QUERY_ROWS, QUERY_TIMEOUT_SECONDS
from core.datasource import DataSource

# sqlglot logs a warning when it falls back to parsing unknown syntax as
# a Command; the guard rejects Commands anyway, so the warning is noise.
logging.getLogger("sqlglot").setLevel(logging.ERROR)


class SQLGuardError(ValueError):
    """Raised when an LLM-generated query fails validation."""


# Node types that must never appear anywhere in a read-only query.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge,
    exp.Create, exp.Drop, exp.Alter, exp.TruncateTable,
    exp.Command, exp.Pragma, exp.Attach, exp.Detach,
    exp.Transaction, exp.Commit, exp.Rollback,
    exp.Set, exp.Use, exp.Copy, exp.Analyze, exp.LoadData,
)

_ALLOWED_ROOTS: tuple[type[exp.Expression], ...] = (exp.Select, exp.SetOperation)


def _trim(sql: str, dialect: str) -> str:
    """Cut the text after the last real token (trailing `;` and comments)."""
    tokens = [t for t in sqlglot.tokenize(sql, read=dialect) if t.token_type != TokenType.SEMICOLON]
    if not tokens:
        return ""
    return sql[: tokens[-1].end + 1].strip()


def _guard(sql: str, dialect: str = "sqlite") -> str:
    """Layer 1: return the query text to execute, or raise SQLGuardError."""
    if not sql or not sql.strip():
        raise SQLGuardError("empty query")

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
        trimmed = _trim(sql, dialect)
    except (ParseError, TokenError) as e:
        raise SQLGuardError(f"could not parse query: {str(e).splitlines()[0]}") from e

    if not statements or not trimmed:
        raise SQLGuardError("empty query after removing comments")
    if len(statements) > 1:
        raise SQLGuardError("multiple statements are not allowed")

    root = statements[0]
    if not isinstance(root, _ALLOWED_ROOTS):
        raise SQLGuardError(f"only SELECT queries are allowed (got {root.key.upper()})")

    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SQLGuardError(f"forbidden operation in query: {node.key.upper()}")

    return trimmed


# ── Layer 2: engine-level read-only ─────────────────────────────────────

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
_DENIED_FUNCTIONS = {"load_extension", "fts3_tokenizer", "readfile", "writefile", "edit"}


def _read_only_authorizer(action, arg1, arg2, db_name, trigger):
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    # For SQLITE_FUNCTION, arg2 is the function name.
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


class Database:
    """A SQLite database the Coder queries: in-memory uploads or a file."""

    dialect = "sqlite"

    def __init__(self, conn: Optional[sqlite3.Connection] = None, name: str = "workspace"):
        self.conn = conn or sqlite3.connect(":memory:", check_same_thread=False)
        self.name = name
        self.table_name = DEFAULT_TABLE_NAME  # primary table (first loaded)
        self.timeout_seconds = QUERY_TIMEOUT_SECONDS
        self._datasource: Optional[DataSource] = None
        self._loaded_any = False

    @classmethod
    def from_sqlite(cls, path: str, name: Optional[str] = None) -> "Database":
        """Open an existing SQLite file read-only (e.g. a Spider/BIRD db)."""
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        # Spider/BIRD contain non-UTF-8 text in a few databases.
        conn.text_factory = lambda b: b.decode(errors="replace")
        db = cls(conn=conn, name=name or path)
        tables = db.datasource().table_names
        if tables:
            db.table_name = tables[0]
        return db

    def load_dataframe(self, df: pd.DataFrame, table_name: Optional[str] = None):
        """Load a DataFrame as a table. The first table loaded is the primary one."""
        name = table_name or self.table_name
        if not self._loaded_any:
            self.table_name = name
            self._loaded_any = True
        df.to_sql(name, self.conn, if_exists="replace", index=False)
        self._datasource = None

    def datasource(self) -> DataSource:
        """Describe every table currently loaded (cached until the next load)."""
        if self._datasource is None:
            self._datasource = DataSource.from_sqlite_conn(self.conn, name=self.name)
        return self._datasource

    def execute_query(self, sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute a read-only SQL query and return (result, error).

        Errors are returned as strings (not raised) so the Coder agent can
        feed them back to the LLM and self-correct on retry.
        """
        try:
            safe_sql = _guard(sql, self.dialect)
        except SQLGuardError as e:
            return None, f"SQLGuardError: {e}"

        deadline = time.monotonic() + self.timeout_seconds
        timed_out = False

        def _progress():
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1  # non-zero aborts the statement
            return 0

        self.conn.set_authorizer(_read_only_authorizer)
        self.conn.set_progress_handler(_progress, 10_000)
        try:
            cur = self.conn.execute(safe_sql)
            columns = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(MAX_QUERY_ROWS)
            cur.close()
            return pd.DataFrame.from_records(rows, columns=columns), None
        except sqlite3.DatabaseError as e:
            if timed_out:
                return None, f"QueryTimeout: query exceeded {self.timeout_seconds}s and was interrupted"
            return None, str(e)
        except Exception as e:
            return None, str(e)
        finally:
            self.conn.set_authorizer(None)
            self.conn.set_progress_handler(None, 0)

    def close(self):
        """Close the database connection."""
        self.conn.close()
