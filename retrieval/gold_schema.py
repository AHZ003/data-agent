"""Tables and columns a gold SQL query uses, resolved against a schema.

Used to measure schema-linking recall without any LLM: did the pruned
schema keep every column the gold query touches? Aliases are resolved
per scope (T1 can mean different tables in different subqueries), and
unqualified columns are attributed to whichever FROM table has them.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from core.datasource import DataSource


def gold_items(sql: str, ds: DataSource, dialect: str = "sqlite") -> tuple[set[str], set[tuple[str, str]]]:
    """(tables, (table, column) pairs), all lower-cased."""
    cols_of = {t.name.lower(): {c.name.lower() for c in t.columns} for t in ds.tables}
    tables: set[str] = set()
    columns: set[tuple[str, str]] = set()
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return tables, columns

    for scope in traverse_scope(tree):
        sources = {}
        for alias, src in scope.sources.items():
            if isinstance(src, exp.Table) and src.name.lower() in cols_of:
                sources[alias.lower()] = src.name.lower()
                tables.add(src.name.lower())
        for col in scope.columns:
            name = col.name.lower()
            if col.table:
                t = sources.get(col.table.lower())
                if t and name in cols_of[t]:
                    columns.add((t, name))
            else:
                owners = [t for t in sources.values() if name in cols_of[t]]
                if len(owners) >= 1:
                    columns.add((owners[0], name))
    return tables, columns
