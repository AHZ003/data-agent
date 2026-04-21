"""SQLite database connection and query execution.

Safety model
------------
The Coder agent emits SQL that runs against user data. The LLM is not
trusted to stay read-only on its own — a prompt-injected CSV or a
creative rephrasing can steer it toward destructive statements. This
module enforces a defense-in-depth allow-list *before* handing the
query to SQLite:

  1. The statement must parse to exactly one SELECT or WITH ... SELECT.
  2. Blocked keywords (DROP, DELETE, UPDATE, INSERT, ATTACH, PRAGMA,
     REPLACE, CREATE, ALTER, TRUNCATE, VACUUM) cause a SQLGuardError
     before execution.
  3. Results are capped at MAX_QUERY_ROWS.

Tests live in tests/test_database_guard.py.
"""

import re
import sqlite3
import pandas as pd
from typing import Optional, Tuple

from config import DEFAULT_TABLE_NAME, MAX_QUERY_ROWS


class SQLGuardError(ValueError):
    """Raised when an LLM-generated query fails the allow-list check."""


_BLOCKED_KEYWORDS = (
    "drop", "delete", "update", "insert", "replace",
    "attach", "detach", "pragma", "create", "alter",
    "truncate", "vacuum", "reindex", "exec",
)

_ALLOWED_PREFIXES = ("select", "with")


def _strip_comments(sql: str) -> str:
    # Remove -- line comments and /* block */ comments.
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql


def _guard(sql: str) -> str:
    """Return a normalized SELECT, or raise SQLGuardError."""
    if not sql or not sql.strip():
        raise SQLGuardError("empty query")

    cleaned = _strip_comments(sql).strip().rstrip(";").strip()
    if not cleaned:
        raise SQLGuardError("empty query after stripping comments")

    # Reject multi-statements — any remaining semicolon means 2+ statements.
    if ";" in cleaned:
        raise SQLGuardError("multiple statements are not allowed")

    head = cleaned.split(None, 1)[0].lower()
    if head not in _ALLOWED_PREFIXES:
        raise SQLGuardError(f"only SELECT/WITH queries are allowed (got '{head}')")

    # Word-boundary keyword scan so "DROPPING" inside a string literal
    # doesn't false-positive, but "DROP TABLE" does.
    lowered = cleaned.lower()
    # Mask single-quoted string literals so keywords inside strings pass.
    masked = re.sub(r"'[^']*'", "''", lowered)
    for kw in _BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", masked):
            raise SQLGuardError(f"blocked keyword: {kw.upper()}")

    return cleaned


class Database:
    """Manages an in-memory SQLite database for uploaded data."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.table_name = DEFAULT_TABLE_NAME

    def load_dataframe(self, df: pd.DataFrame, table_name: Optional[str] = None):
        """Load a DataFrame into the SQLite database."""
        if table_name:
            self.table_name = table_name
        # Clean column names for SQL compatibility
        clean_df = df.copy()
        clean_df.to_sql(self.table_name, self.conn, if_exists="replace", index=False)

    def execute_query(self, sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute a read-only SQL query and return results or error.

        Runs the query through the SQL allow-list first (see module
        docstring). Any mutation, DDL, or multi-statement input is
        rejected with a clear error that the Coder agent can use to
        self-correct on retry.
        """
        try:
            safe_sql = _guard(sql)
        except SQLGuardError as e:
            return None, f"SQLGuardError: {e}"
        try:
            result = pd.read_sql(safe_sql, self.conn)
            if len(result) > MAX_QUERY_ROWS:
                result = result.head(MAX_QUERY_ROWS)
            return result, None
        except Exception as e:
            return None, str(e)

    def close(self):
        """Close the database connection."""
        self.conn.close()
