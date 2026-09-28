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


# ── Former false positives of the regex guard ────────────────────────────

@pytest.mark.parametrize(
    "ok",
    [
        "SELECT REPLACE(b, '_', ' ') FROM t",       # REPLACE() the function
        'SELECT "update" FROM t',                    # keyword as quoted column
        "SELECT b FROM t WHERE b = 'it''s; fine'",   # escaped quote + semicolon
        "SELECT b FROM t WHERE b = 'x -- not a comment'",
        "SELECT a FROM t UNION SELECT a FROM t",
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 5) SELECT i FROM n",
    ],
)
def test_guard_allows_valid_selects(ok):
    _guard(ok)


@pytest.mark.parametrize(
    "bad",
    [
        "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d",
        "DETACH DATABASE x",
        "BEGIN",
        "SELECT 1 /* ; */ ; DROP TABLE t",
        "CREATE TEMP TABLE x AS SELECT 1",
    ],
)
def test_guard_blocks_nested_and_obfuscated(bad):
    with pytest.raises(SQLGuardError):
        _guard(bad)


def test_db_runs_former_false_positives():
    d = Database()
    d.load_dataframe(pd.DataFrame({"region": ["a_b"], "update": [1], "note": ["it's; fine"]}), "data")
    for sql in (
        "SELECT REPLACE(region, '_', ' ') AS r FROM data",
        'SELECT "update" FROM data',
        "SELECT note FROM data WHERE note = 'it''s; fine'",
    ):
        result, err = d.execute_query(sql)
        assert err is None, (sql, err)
        assert len(result) == 1


# ── Layer 2: authorizer blocks writes even if layer 1 is bypassed ────────

def test_authorizer_blocks_delete_when_guard_bypassed(db, monkeypatch):
    import core.database as dbmod
    monkeypatch.setattr(dbmod, "_guard", lambda sql, dialect="sqlite": sql)
    result, err = db.execute_query("DELETE FROM t")
    assert result is None
    assert "not authorized" in err
    count, _ = db.execute_query("SELECT COUNT(*) AS n FROM t")
    assert int(count["n"].iloc[0]) == 3


def test_authorizer_blocks_pragma_and_attach_when_guard_bypassed(db, monkeypatch):
    import core.database as dbmod
    monkeypatch.setattr(dbmod, "_guard", lambda sql, dialect="sqlite": sql)
    for sql in ("PRAGMA table_info(t)", "ATTACH DATABASE ':memory:' AS x"):
        result, err = db.execute_query(sql)
        assert result is None and err, sql


def test_authorizer_is_removed_after_query(db):
    db.execute_query("SELECT 1")
    # Loading more data must still work once the guarded query is done.
    db.load_dataframe(pd.DataFrame({"z": [1]}), "t2")
    result, err = db.execute_query("SELECT z FROM t2")
    assert err is None and list(result["z"]) == [1]


# ── Layer 3: timeout ─────────────────────────────────────────────────────

def test_runaway_recursive_cte_times_out(db):
    db.timeout_seconds = 0.2
    result, err = db.execute_query(
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT MAX(i) FROM n"
    )
    assert result is None
    assert err.startswith("QueryTimeout")


def test_result_rows_are_capped(db):
    from config import MAX_QUERY_ROWS
    result, err = db.execute_query(
        f"WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < {MAX_QUERY_ROWS + 50}) "
        "SELECT i FROM n"
    )
    assert err is None
    assert len(result) == MAX_QUERY_ROWS


# ── Property tests ───────────────────────────────────────────────────────

from hypothesis import given, settings, strategies as st  # noqa: E402

_COLS = ["a", "b"]
_col = st.sampled_from(_COLS)
_select = st.builds(
    lambda cols, where, order, limit: (
        f"SELECT {', '.join(cols)} FROM t"
        + (f" WHERE a {where[0]} {where[1]}" if where else "")
        + (f" ORDER BY {order}" if order else "")
        + (f" LIMIT {limit}" if limit is not None else "")
    ),
    st.lists(_col, min_size=1, max_size=2),
    st.none() | st.tuples(st.sampled_from(["=", "<", ">", "<=", ">=", "!="]), st.integers(-5, 5)),
    st.none() | _col,
    st.none() | st.integers(0, 10),
)
_literal = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20).map(
    lambda s: "'" + s.replace("'", "''") + "'"
)
_dml = st.one_of(
    st.builds(lambda v: f"INSERT INTO t (b) VALUES ({v})", _literal),
    st.builds(lambda v: f"UPDATE t SET b = {v}", _literal),
    st.builds(lambda v: f"DELETE FROM t WHERE b = {v}", _literal),
    st.just("DROP TABLE t"),
    st.builds(lambda s: f"{s}; DROP TABLE t", _select),
)


@settings(max_examples=100, deadline=None)
@given(_select)
def test_random_selects_never_raise(sql):
    _guard(sql)


@settings(max_examples=100, deadline=None)
@given(_literal)
def test_string_literals_never_trip_the_guard(literal):
    _guard(f"SELECT b FROM t WHERE b = {literal}")


@settings(max_examples=100, deadline=None)
@given(_dml)
def test_random_dml_always_rejected(sql):
    with pytest.raises(SQLGuardError):
        _guard(sql)
