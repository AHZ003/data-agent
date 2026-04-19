"""SQLite-backed persistence for workspaces, datasets, and analyses."""

import json
import os
import pickle
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


DB_PATH = os.environ.get(
    "DATAAGENT_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db_data", "dataagent.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    name TEXT NOT NULL,
    original_filename TEXT,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    data_blob BLOB NOT NULL,
    schema_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    question TEXT NOT NULL,
    sql_query TEXT,
    result_blob BLOB,
    chart_json TEXT,
    prediction_chart_json TEXT,
    prediction_json TEXT,
    narrative TEXT,
    validation_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_datasets_workspace ON datasets(workspace_id);
CREATE INDEX IF NOT EXISTS idx_analyses_workspace ON analyses(workspace_id);
CREATE INDEX IF NOT EXISTS idx_analyses_dataset ON analyses(dataset_id);
"""


@dataclass
class Workspace:
    id: str
    name: str
    created_at: str


@dataclass
class DatasetRow:
    id: str
    workspace_id: str
    name: str
    original_filename: Optional[str]
    row_count: int
    column_count: int
    created_at: str


@dataclass
class AnalysisRow:
    id: str
    workspace_id: str
    dataset_id: str
    question: str
    created_at: str


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def init_db() -> None:
    """Create the DB file + tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _conn() as c:
        c.executescript(_SCHEMA)


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# --- Workspaces ---

def create_workspace(name: str) -> Workspace:
    wid = _new_id()
    with _conn() as c:
        c.execute("INSERT INTO workspaces (id, name) VALUES (?, ?)", (wid, name))
    return get_workspace(wid)


def get_workspace(workspace_id: str) -> Optional[Workspace]:
    with _conn() as c:
        row = c.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
    return Workspace(**dict(row)) if row else None


def list_workspaces() -> list[Workspace]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM workspaces ORDER BY created_at DESC").fetchall()
    return [Workspace(**dict(r)) for r in rows]


def delete_workspace(workspace_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))


def ensure_default_workspace() -> Workspace:
    ws = list_workspaces()
    if ws:
        return ws[0]
    return create_workspace("Default")


# --- Datasets ---

def save_dataset(
    workspace_id: str,
    name: str,
    df: pd.DataFrame,
    schema_dict: dict,
    original_filename: Optional[str] = None,
) -> str:
    dsid = _new_id()
    blob = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
    with _conn() as c:
        c.execute(
            """INSERT INTO datasets
               (id, workspace_id, name, original_filename, row_count, column_count, data_blob, schema_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (dsid, workspace_id, name, original_filename, len(df), len(df.columns),
             blob, json.dumps(schema_dict, default=str)),
        )
    return dsid


def load_dataset(dataset_id: str) -> Optional[dict[str, Any]]:
    with _conn() as c:
        row = c.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "workspace_id": row["workspace_id"],
        "name": row["name"],
        "original_filename": row["original_filename"],
        "df": pickle.loads(row["data_blob"]),
        "schema": json.loads(row["schema_json"]),
        "created_at": row["created_at"],
    }


def list_datasets(workspace_id: str) -> list[DatasetRow]:
    with _conn() as c:
        rows = c.execute(
            """SELECT id, workspace_id, name, original_filename, row_count, column_count, created_at
               FROM datasets WHERE workspace_id = ? ORDER BY created_at DESC""",
            (workspace_id,),
        ).fetchall()
    return [DatasetRow(**dict(r)) for r in rows]


def delete_dataset(dataset_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))


# --- Analyses ---

def save_analysis(
    workspace_id: str,
    dataset_id: str,
    analysis: dict[str, Any],
) -> str:
    aid = _new_id()
    result_df = analysis.get("result_df")
    result_blob = pickle.dumps(result_df, protocol=pickle.HIGHEST_PROTOCOL) if result_df is not None else None

    chart = analysis.get("chart")
    chart_json = chart.to_json() if isinstance(chart, go.Figure) else None

    pchart = analysis.get("prediction_chart")
    pchart_json = pchart.to_json() if isinstance(pchart, go.Figure) else None

    with _conn() as c:
        c.execute(
            """INSERT INTO analyses
               (id, workspace_id, dataset_id, question, sql_query, result_blob,
                chart_json, prediction_chart_json, prediction_json, narrative, validation_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                aid, workspace_id, dataset_id,
                analysis.get("question", ""),
                analysis.get("sql_query"),
                result_blob,
                chart_json,
                pchart_json,
                json.dumps(analysis.get("prediction"), default=str) if analysis.get("prediction") else None,
                analysis.get("narrative"),
                json.dumps(analysis.get("validation"), default=str) if analysis.get("validation") else None,
            ),
        )
    return aid


def load_analysis(analysis_id: str) -> Optional[dict[str, Any]]:
    with _conn() as c:
        row = c.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "workspace_id": row["workspace_id"],
        "dataset_id": row["dataset_id"],
        "question": row["question"],
        "sql_query": row["sql_query"],
        "result_df": pickle.loads(row["result_blob"]) if row["result_blob"] else None,
        "chart": pio.from_json(row["chart_json"]) if row["chart_json"] else None,
        "prediction_chart": pio.from_json(row["prediction_chart_json"]) if row["prediction_chart_json"] else None,
        "prediction": json.loads(row["prediction_json"]) if row["prediction_json"] else None,
        "narrative": row["narrative"],
        "validation": json.loads(row["validation_json"]) if row["validation_json"] else None,
        "created_at": row["created_at"],
    }


def list_analyses(workspace_id: str, dataset_id: Optional[str] = None) -> list[AnalysisRow]:
    with _conn() as c:
        if dataset_id:
            rows = c.execute(
                """SELECT id, workspace_id, dataset_id, question, created_at
                   FROM analyses WHERE dataset_id = ? ORDER BY created_at DESC""",
                (dataset_id,),
            ).fetchall()
        else:
            rows = c.execute(
                """SELECT id, workspace_id, dataset_id, question, created_at
                   FROM analyses WHERE workspace_id = ? ORDER BY created_at DESC""",
                (workspace_id,),
            ).fetchall()
    return [AnalysisRow(**dict(r)) for r in rows]
