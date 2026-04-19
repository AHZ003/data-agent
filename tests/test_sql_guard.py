"""Tests for the SQL allow-list that sits in front of Database.execute_query."""

import pandas as pd
import pytest

from core.database import Database, SQLGuardError, _guard


# ── _guard unit tests ────────────────────────────────────────────────────

def test_guard_allows_select():
    assert _guard("SELECT * FROM t").lower().startswith("select")


def test_guard_allows_with_cte():
    assert _guard("WITH x AS (SELECT 1) SELECT * FROM x").lower().startswith("with")


def test_guard_strips_trailing_semicolon():
    assert _guard("SELECT 1;") == "SELECT 1"


def test_guard_strips_comments():
    out = _guard("SELECT 1 -- sneaky comment\n")
    assert "--" not in out


@pytest.mark.parametrize(
    "bad",
    [
        "DROP TABLE t",
        "DELETE FROM t WHERE 1=1",
        "UPDATE t SET a=1",
        "INSERT INTO t VALUES (1)",
        "REPLACE INTO t VALUES (1)",
        "ALTER TABLE t ADD COLUMN x INT",
        "TRUNCATE t",
        "PRAGMA table_info(t)",
        "ATTACH DATABASE '/tmp/x' AS x",
        "VACUUM",
    ],
)
def test_guard_blocks_mutations(bad):
    with pytest.raises(SQLGuardError):
        _guard(bad)


def test_guard_blocks_multi_statement():
    with pytest.raises(SQLGuardError):
        _guard("SELECT 1; DROP TABLE t")


def test_guard_allows_keyword_inside_string_literal():
    # A literal like 'drop the ball' in data should not trip the guard.
    out = _guard("SELECT * FROM t WHERE note = 'drop the ball'")
    assert "where" in out.lower()


def test_guard_rejects_empty():
    with pytest.raises(SQLGuardError):
        _guard("")
    with pytest.raises(SQLGuardError):
        _guard("   ")
    with pytest.raises(SQLGuardError):
        _guard("-- only a comment")


# ── Database integration ─────────────────────────────────────────────────

@pytest.fixture
def db():
    d = Database()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    d.load_dataframe(df, "t")
    return d


def test_db_executes_select(db):
    result, err = db.execute_query("SELECT a FROM t ORDER BY a")
    assert err is None
    assert list(result["a"]) == [1, 2, 3]


def test_db_rejects_drop(db):
    result, err = db.execute_query("DROP TABLE t")
    assert result is None
    assert "SQLGuardError" in err
    # Ensure t still exists and still has data.
    result2, err2 = db.execute_query("SELECT COUNT(*) AS n FROM t")
    assert err2 is None
    assert int(result2["n"].iloc[0]) == 3


def test_db_rejects_multi_statement(db):
    result, err = db.execute_query("SELECT 1; DELETE FROM t")
    assert result is None
    assert "SQLGuardError" in err
