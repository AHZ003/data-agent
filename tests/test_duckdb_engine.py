"""DuckDB engine: loading, introspection, guard, file-access lockdown, timeout."""

import pandas as pd
import pytest

from core.duckdb_engine import DuckDBEngine
from retrieval.values import build


@pytest.fixture
def eng(tmp_path):
    csv = tmp_path / "orders.csv"
    pd.DataFrame({"order_id": [1, 2, 3], "region": ["West", "East", "West"],
                  "amount": [5.0, 7.5, 3.0]}).to_csv(csv, index=False)
    e = DuckDBEngine(name="shop")
    e.load_file(str(csv), "orders")
    e.load_dataframe(pd.DataFrame({"region": ["West", "East"], "manager": ["Ada", "Lin"]}), "regions")
    return e


def test_introspection_and_prompt(eng):
    ds = eng.datasource()
    assert ds.dialect == "duckdb"
    assert set(ds.table_names) == {"orders", "regions"}
    region = ds.table("orders").column("region")
    assert region.distinct_count == 2 and region.sample_values[0] == "West"
    assert 'Table "orders" (3 rows)' in ds.to_prompt()


def test_guarded_join_query(eng):
    df, err = eng.execute_query(
        'SELECT r.manager, SUM(o.amount) AS total FROM orders o JOIN regions r ON o.region = r.region '
        "GROUP BY r.manager ORDER BY total DESC")
    assert err is None
    assert df.to_dict("records") == [{"manager": "Lin", "total": 7.5}, {"manager": "Ada", "total": 8.0}][::-1] \
        or df.to_dict("records") == [{"manager": "Ada", "total": 8.0}, {"manager": "Lin", "total": 7.5}]


@pytest.mark.parametrize("sql", [
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM '/etc/passwd'",
    "SELECT * FROM glob('*')",
    "COPY orders TO '/tmp/x.csv'",
    "DELETE FROM orders",
])
def test_guard_blocks_file_access_and_writes(eng, sql):
    df, err = eng.execute_query(sql)
    assert df is None and "SQLGuardError" in err


def test_engine_blocks_file_reads_even_if_guard_is_bypassed(eng, monkeypatch, tmp_path):
    import core.duckdb_engine as mod
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n")
    monkeypatch.setattr(mod, "_guard", lambda sql, dialect: sql)
    df, err = eng.execute_query(f"SELECT * FROM read_csv('{secret}')")
    assert df is None and err
    df, err = eng.execute_query("SET enable_external_access = true")
    assert df is None and err  # configuration is locked


def test_cannot_load_after_lock(eng):
    eng.execute_query("SELECT 1")
    with pytest.raises(RuntimeError):
        eng.load_dataframe(pd.DataFrame({"x": [1]}), "late")


def test_timeout_interrupts_long_query(eng):
    eng.timeout_seconds = 0.3
    df, err = eng.execute_query("SELECT COUNT(*) FROM range(10000000000) a, range(1000) b")
    assert df is None and err.startswith("QueryTimeout")


def test_value_index_works_on_duckdb(eng):
    idx = build(eng)
    assert any(m.value == "West" and m.column == "region" for m in idx.match("sales in the west"))
