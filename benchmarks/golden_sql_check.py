"""Validate the golden SQL set: gold runs, is non-empty, and has no LIMIT ties.

    python -m benchmarks.golden_sql_check

A question like "Which artist has the most albums?" has no single right
answer if two artists tie for first; a correct prediction that picks
the other one would be scored wrong. For every ORDER BY ... LIMIT gold
query we drop the LIMIT, project the ORDER BY keys, and check that the
last row kept differs from the first row dropped. Window-function
rankings (RANK() ... = 1) are checked for one row per partition.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import sqlglot
import yaml
from sqlglot import exp

ROOT = Path(__file__).resolve().parent.parent
YAML = ROOT / "benchmarks" / "golden_sql.yaml"


def load(path: Path = YAML) -> tuple[Path, list[dict]]:
    data = yaml.safe_load(path.read_text())
    return ROOT / data["db"], data["cases"]


def limit_tie(conn: sqlite3.Connection, sql: str) -> str | None:
    """Return a description of a tie at the LIMIT boundary, or None."""
    tree = sqlglot.parse_one(sql, read="sqlite")
    limit, order = tree.args.get("limit"), tree.args.get("order")
    if not isinstance(tree, exp.Select) or limit is None or order is None:
        return None
    k = int(limit.expression.name)
    if k < 0:
        return None
    offset = int(tree.args["offset"].expression.name) if tree.args.get("offset") else 0
    probe = tree.copy()
    probe.set("limit", None)
    probe.set("offset", None)
    # ORDER BY may name a select alias, which SQLite can't reuse inside
    # another select expression; project the aliased expression instead.
    aliases = {a.alias: a.this for a in tree.expressions if isinstance(a, exp.Alias)}
    for i, o in enumerate(order.expressions):
        key = o.this
        if isinstance(key, exp.Column) and not key.table and key.name in aliases:
            key = aliases[key.name]
        probe = probe.select(exp.alias_(key.copy(), f"__ok{i}"), append=True)
    rows = conn.execute(probe.sql(dialect="sqlite")).fetchall()
    n_keys = len(order.expressions)
    last, nxt = offset + k - 1, offset + k
    if nxt < len(rows) and rows[last][-n_keys:] == rows[nxt][-n_keys:]:
        return f"rows {last} and {nxt} tie on the ORDER BY key {rows[last][-n_keys:]}"
    return None


def check(path: Path = YAML) -> list[str]:
    db, cases = load(path)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    problems = []
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        problems.append("duplicate case ids")
    for c in cases:
        try:
            rows = conn.execute(c["gold_sql"]).fetchall()
        except Exception as e:
            problems.append(f"{c['id']}: gold SQL fails: {e}")
            continue
        if not rows or all(v is None for r in rows for v in r):
            problems.append(f"{c['id']}: gold result is empty")
        tie = limit_tie(conn, c["gold_sql"])
        if tie:
            problems.append(f"{c['id']}: {tie}")
        if "RANK()" in c["gold_sql"].upper():
            keys = [r[0] for r in rows]
            if len(keys) != len(set(keys)):
                problems.append(f"{c['id']}: RANK() = 1 returns ties within a partition")
    return problems


if __name__ == "__main__":
    problems = check()
    _, cases = load()
    for p in problems:
        print("PROBLEM", p)
    print(f"{len(cases)} cases, {len(problems)} problems")
    sys.exit(1 if problems else 0)
