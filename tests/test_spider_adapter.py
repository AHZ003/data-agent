"""Unit tests for the Spider benchmark adapter's utility functions.

These tests don't need the actual Spider dataset or an API key — they
exercise the SQL extraction, schema formatting, result matching, and
summary computation logic.
"""

import sqlite3
from pathlib import Path

from benchmarks.spider_eval import (
    SpiderResult,
    _extract_sql,
    _get_spider_schema,
    _results_match,
    _schema_to_prompt_context,
    _build_spider_prompt,
    summarize,
    render_report,
)


# ── SQL extraction ───────────────────────────────────────────────────────

def test_extract_sql_code_block():
    assert _extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"


def test_extract_sql_bare():
    assert _extract_sql("SELECT * FROM t WHERE x = 1") == "SELECT * FROM t WHERE x = 1"


def test_extract_sql_with_preamble():
    assert _extract_sql("The SQL query:\nSELECT 1") == "SELECT 1"


# ── Schema helpers ───────────────────────────────────────────────────────

def test_get_spider_schema_from_sqlite(tmp_path):
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE city (id INTEGER PRIMARY KEY, name TEXT, pop REAL)")
    conn.execute("CREATE TABLE country (code TEXT PRIMARY KEY, name TEXT)")
    conn.close()

    schema = _get_spider_schema(db_path)
    assert "city" in schema
    assert "country" in schema
    assert len(schema["city"]) == 3
    assert schema["city"][0]["name"] == "id"
    assert schema["city"][0]["pk"] is True


def test_schema_to_prompt_context_format():
    tables = {
        "employees": [
            {"name": "id", "dtype": "INTEGER", "pk": True},
            {"name": "name", "dtype": "TEXT", "pk": False},
        ]
    }
    text = _schema_to_prompt_context(tables)
    assert "Table: employees" in text
    assert "id (INTEGER) [PK]" in text
    assert "name (TEXT)" in text
    assert "[PK]" not in text.split("name (TEXT)")[1] if "name (TEXT)" in text else True


def test_build_spider_prompt_contains_key_parts():
    prompt = _build_spider_prompt(
        "What is the population?",
        "Table: city\n  name (TEXT)\n  pop (REAL)",
        ["city"],
    )
    assert "What is the population?" in prompt
    assert "city" in prompt
    assert "SQLite" in prompt


# ── Result matching ──────────────────────────────────────────────────────

def test_results_match_identical():
    rows = [("a", "1"), ("b", "2")]
    assert _results_match(rows, rows) is True


def test_results_match_different():
    assert _results_match([("a",)], [("b",)]) is False


def test_results_match_none():
    assert _results_match(None, [("a",)]) is False
    assert _results_match([("a",)], None) is False


# ── Summary + report ─────────────────────────────────────────────────────

def test_summarize_computes_accuracy():
    results = [
        SpiderResult(0, "db1", "q1", "SELECT 1", "SELECT 1", match=True, duration_s=1.0),
        SpiderResult(1, "db1", "q2", "SELECT 2", "SELECT 3", match=False, duration_s=2.0),
        SpiderResult(2, "db2", "q3", "SELECT 4", "", match=False, error="429", duration_s=0.5),
    ]
    s = summarize(results, "test-model")
    assert s.n_total == 3
    assert s.n_correct == 1
    assert s.n_error == 1
    assert s.execution_accuracy == round(1 / 3, 4)
    assert s.model == "test-model"


def test_render_report_is_markdown():
    results = [
        SpiderResult(0, "db1", "q1", "SELECT 1", "SELECT 1", match=True, duration_s=1.0),
    ]
    s = summarize(results, "m")
    md = render_report(results, s)
    assert "# DataAgent" in md
    assert "100%" in md
