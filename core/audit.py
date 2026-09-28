"""Audit log: who asked what, which SQL ran, what it scanned, what PII it touched.

One row per completed API run in a SQLite file (DATAAGENT_AUDIT_PATH,
default outputs/audit.sqlite). PII columns touched are found by parsing
the executed SQL (alias-aware, retrieval/gold_schema.py) and intersecting
with the PII columns detected in the datasource (core/pii.py).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
_lock = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    run_id TEXT NOT NULL,
    api_key TEXT NOT NULL,
    question TEXT NOT NULL,
    sql TEXT,
    engine TEXT,
    datasource TEXT,
    bytes_scanned INTEGER,
    pii_columns_touched TEXT NOT NULL,
    status TEXT NOT NULL
)
"""


def _path() -> Path:
    return Path(os.getenv("DATAAGENT_AUDIT_PATH", str(ROOT / "outputs" / "audit.sqlite")))


def _conn() -> sqlite3.Connection:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def pii_columns_touched(sql: Optional[str], engine) -> list[str]:
    if not sql:
        return []
    from core.pii import detect
    from retrieval.gold_schema import gold_items

    ds = engine.datasource()
    report = detect(ds)
    _, cols = gold_items(sql, ds, dialect=ds.dialect)
    pii_cols = {(t.lower(), c.lower()) for (t, c) in report.columns}
    return sorted(f"{t}.{c}" for t, c in cols & pii_cols)


def record(run_id: str, api_key: str, question: str, sql: Optional[str], engine, status: str) -> None:
    bytes_scanned = None
    if sql and getattr(engine, "dialect", "") == "bigquery":
        try:
            bytes_scanned = engine.dry_run(sql).bytes_processed
        except Exception:
            pass
    row = (time.time(), run_id, api_key, question, sql, type(engine).__name__, getattr(engine, "name", ""),
           bytes_scanned, json.dumps(pii_columns_touched(sql, engine)), status)
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO audit (ts, run_id, api_key, question, sql, engine, datasource, bytes_scanned, "
            "pii_columns_touched, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)


def recent(limit: int = 100, api_key: Optional[str] = None) -> list[dict]:
    q = "SELECT * FROM audit" + (" WHERE api_key = ?" if api_key else "") + " ORDER BY id DESC LIMIT ?"
    params = ((api_key,) if api_key else ()) + (limit,)
    with _lock, _conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(q, params)]
    for r in rows:
        r["pii_columns_touched"] = json.loads(r["pii_columns_touched"])
    return rows
