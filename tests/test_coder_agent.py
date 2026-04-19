"""Tests for the Coder Agent (non-API parts)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from core.database import Database
from agents.coder_agent import _extract_sql


def test_extract_sql_from_code_block():
    response = """Here's the query:
```sql
SELECT Category, SUM(Revenue) FROM sales GROUP BY Category
```
"""
    sql = _extract_sql(response)
    assert "SELECT" in sql
    assert "GROUP BY" in sql


def test_extract_sql_plain():
    response = "SELECT COUNT(*) FROM sales"
    sql = _extract_sql(response)
    assert sql == "SELECT COUNT(*) FROM sales"


def test_database_load_and_query():
    db = Database()
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "score": [90, 85, 95],
    })
    db.load_dataframe(df, "students")

    result, error = db.execute_query("SELECT name, score FROM students WHERE score > 88")
    assert error is None
    assert len(result) == 2
    assert "Alice" in result["name"].values


def test_database_query_error():
    db = Database()
    df = pd.DataFrame({"x": [1, 2, 3]})
    db.load_dataframe(df, "test")

    result, error = db.execute_query("SELECT nonexistent FROM test")
    assert result is None
    assert error is not None


if __name__ == "__main__":
    test_extract_sql_from_code_block()
    print("✓ test_extract_sql_from_code_block passed")
    test_extract_sql_plain()
    print("✓ test_extract_sql_plain passed")
    test_database_load_and_query()
    print("✓ test_database_load_and_query passed")
    test_database_query_error()
    print("✓ test_database_query_error passed")
    print("\nAll coder agent tests passed!")
