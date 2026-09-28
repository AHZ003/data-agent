"""Tests for DataSource introspection and multi-table prompt rendering."""

import sqlite3

import pandas as pd
import pytest

from agents.coder_agent import _build_sql_prompt
from core.database import Database
from core.datasource import DataSource


@pytest.fixture
def two_table_db():
    db = Database()
    db.load_dataframe(pd.DataFrame({
        "customer_id": [1, 2, 3],
        "region": ["West", "East", "West"],
    }), "customers")
    db.load_dataframe(pd.DataFrame({
        "order_id": [10, 11, 12, 13],
        "customer_id": [1, 1, 2, 3],
        "amount": [5.0, 7.5, 3.0, 9.0],
    }), "orders")
    return db


def test_first_loaded_table_is_primary(two_table_db):
    assert two_table_db.table_name == "customers"


def test_datasource_lists_all_tables(two_table_db):
    ds = two_table_db.datasource()
    assert ds.table_names == ["customers", "orders"]
    region = ds.table("customers").column("region")
    assert region.distinct_count == 2
    assert region.sample_values[0] == "West"  # most frequent first
    assert ds.table("orders").row_count == 4


def test_datasource_cache_invalidated_on_load(two_table_db):
    assert len(two_table_db.datasource().tables) == 2
    two_table_db.load_dataframe(pd.DataFrame({"x": [1]}), "extra")
    assert "extra" in two_table_db.datasource().table_names


def test_join_across_uploaded_tables(two_table_db):
    result, err = two_table_db.execute_query(
        'SELECT c."region", SUM(o."amount") AS total '
        'FROM "orders" o JOIN "customers" c ON o."customer_id" = c."customer_id" '
        'GROUP BY c."region" ORDER BY total DESC'
    )
    assert err is None
    assert result.to_dict("records") == [
        {"region": "West", "total": 21.5},
        {"region": "East", "total": 3.0},
    ]


def test_from_sqlite_reads_keys(tmp_path):
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            total REAL
        );
        INSERT INTO customers VALUES (1, 'Ada'), (2, 'Lin');
        INSERT INTO orders VALUES (1, 1, 9.5);
    """)
    conn.close()

    ds = DataSource.from_sqlite(str(path))
    orders = ds.table("orders")
    assert orders.primary_key == ["id"]
    assert orders.foreign_keys[0].ref_table == "customers"
    assert orders.foreign_keys[0].ref_column == "id"

    text = ds.to_prompt()
    assert 'Table "orders" (1 rows)' in text
    assert '"customer_id" INTEGER -> "customers"."id"' in text
    assert '"id" INTEGER PRIMARY KEY' in text


def test_from_sqlite_database_is_read_only(tmp_path):
    path = tmp_path / "ro.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (a INT)")
    conn.commit()
    conn.close()
    db = Database.from_sqlite(str(path))
    assert db.table_name == "t"
    with pytest.raises(sqlite3.OperationalError):
        db.conn.execute("INSERT INTO t VALUES (1)")


def test_from_dataframes_matches_stored_format():
    ds = DataSource.from_dataframes({
        "events": pd.DataFrame({"day": pd.to_datetime(["2024-01-02"]), "n": [1]})
    })
    day = ds.table("events").column("day")
    # pandas datetimes are stored as text in SQLite — the prompt must show that.
    assert day.sample_values == ["2024-01-02 00:00:00"]


def test_prompt_escapes_quotes_and_truncates_long_values():
    ds = DataSource.from_dataframes({"t": pd.DataFrame({"s": ["it's", "x" * 100]})})
    text = ds.to_prompt()
    assert "'it''s'" in text
    assert "x" * 100 not in text


def test_coder_prompt_renders_every_table(two_table_db):
    prompt = _build_sql_prompt("total by region", two_table_db.datasource())
    assert 'Table "customers"' in prompt
    assert 'Table "orders"' in prompt
    assert "Sqlite" in prompt
    assert "USER QUESTION: total by region" in prompt


def test_composite_primary_key_rendered_on_table_line(tmp_path):
    path = tmp_path / "c.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE pt (a INTEGER, b INTEGER, PRIMARY KEY (a, b))")
    conn.close()
    text = DataSource.from_sqlite(str(path)).to_prompt()
    assert 'Table "pt" (0 rows) PRIMARY KEY ("a", "b")' in text
    assert '"a" INTEGER PRIMARY KEY' not in text
