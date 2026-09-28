"""Tests for the text-to-SQL harness and the Spider adapter.

No Spider download or API key needed: a tiny SQLite database stands in
for a Spider db and `_generate_sql` is patched with canned SQL, so the
real `generate_and_execute` path (guard, execution, self-repair) runs.
"""

import sqlite3
from unittest.mock import patch

import pytest

from agents import coder_agent
from benchmarks.spider_hardness import hardness
from benchmarks.text2sql import (
    Example,
    RunConfig,
    bird_result_eq,
    load_cached,
    cache_path,
    run_examples,
    spider_result_eq,
    stratified_subset,
    summarize,
    render_report,
)


# ── Official result equivalence ──────────────────────────────────────────

def test_spider_eq_ignores_row_order_without_order_by():
    assert spider_result_eq([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], order_matters=False)


def test_spider_eq_respects_order_with_order_by():
    assert not spider_result_eq([(1,), (2,)], [(2,), (1,)], order_matters=True)
    assert spider_result_eq([(1,), (2,)], [(1,), (2,)], order_matters=True)


def test_spider_eq_allows_column_permutation():
    assert spider_result_eq([(1, "a"), (2, "b")], [("a", 1), ("b", 2)], order_matters=False)


def test_spider_eq_is_multiset_not_set():
    assert not spider_result_eq([(1,), (1,), (2,)], [(1,), (2,), (2,)], order_matters=False)


def test_spider_eq_rejects_arity_and_length_mismatch():
    assert not spider_result_eq([(1, 2)], [(1,)], order_matters=False)
    assert not spider_result_eq([(1,)], [(1,), (1,)], order_matters=False)
    assert spider_result_eq([], [], order_matters=False)


def test_bird_eq_is_set_equality():
    assert bird_result_eq([(1,), (1,), (2,)], [(2,), (1,)])
    assert not bird_result_eq([(1,)], [(1,), (3,)])


# ── Official hardness port ───────────────────────────────────────────────

def _sql(**over):
    base = {
        "select": [False, [[0, [0, [0, 1, False], None]]]],
        "from": {"table_units": [["table_unit", 0]], "conds": []},
        "where": [], "groupBy": [], "having": [], "orderBy": [],
        "limit": None, "intersect": None, "union": None, "except": None,
    }
    base.update(over)
    return base


def test_hardness_easy_and_extra():
    assert hardness(_sql()) == "easy"
    nested = _sql()
    assert hardness(_sql(union=nested, intersect=nested, limit=1, orderBy=["asc", []],
                         groupBy=[[0, 1, False]])) == "extra"


# ── End-to-end through the real coder path on a stand-in db ─────────────

@pytest.fixture
def tiny_db(tmp_path):
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE city (id INTEGER PRIMARY KEY, name TEXT, pop INTEGER);
        INSERT INTO city VALUES (1, 'Oslo', 700), (2, 'Bergen', 285), (3, 'Tromso', 77);
    """)
    conn.close()
    return path


def _examples():
    return [
        Example("e1", "shop", "How many cities?", "SELECT count(*) FROM city", "easy"),
        Example("e2", "shop", "Largest city?", "SELECT name FROM city ORDER BY pop DESC LIMIT 1", "medium"),
    ]


def test_run_examples_scores_and_repairs(tiny_db, tmp_path):
    # e1: first attempt errors, repair fixes it. e2: wrong answer.
    replies = iter([
        "SELECT count(*) FROM citty",
        "SELECT count(*) FROM city",
        "SELECT name FROM city ORDER BY pop ASC LIMIT 1",
    ])
    cfg = RunConfig(benchmark="unit", model="m", mode="repair")
    with patch.object(coder_agent, "_generate_sql", side_effect=lambda *a, **k: next(replies)):
        results = run_examples(_examples(), lambda ex: tiny_db, cfg, rule="spider",
                               cache_dir=tmp_path, progress=False)
    assert [r.match for r in results] == [True, False]
    assert results[0].attempts == 2
    s = summarize(results, cfg)
    assert s["ex"] == 0.5 and s["repaired"] == 1
    assert "Execution accuracy" in render_report("t", results, s)


def test_single_mode_does_not_repair(tiny_db, tmp_path):
    replies = iter(["SELECT count(*) FROM citty", "SELECT name FROM city ORDER BY pop DESC LIMIT 1"])
    cfg = RunConfig(benchmark="unit", model="m", mode="single")
    with patch.object(coder_agent, "_generate_sql", side_effect=lambda *a, **k: next(replies)):
        results = run_examples(_examples(), lambda ex: tiny_db, cfg, rule="spider",
                               cache_dir=tmp_path, progress=False)
    assert [r.match for r in results] == [False, True]
    assert results[0].attempts == 1 and results[0].error


def test_cache_makes_reruns_free(tiny_db, tmp_path):
    cfg = RunConfig(benchmark="unit", model="m", mode="single")
    with patch.object(coder_agent, "_generate_sql", return_value="SELECT count(*) FROM city") as gen:
        run_examples(_examples()[:1], lambda ex: tiny_db, cfg, rule="spider", cache_dir=tmp_path, progress=False)
        run_examples(_examples()[:1], lambda ex: tiny_db, cfg, rule="spider", cache_dir=tmp_path, progress=False)
    assert gen.call_count == 1
    assert "e1" in load_cached(cache_path(cfg, tmp_path))


def test_terminal_errors_are_not_cached(tiny_db, tmp_path):
    cfg = RunConfig(benchmark="unit", model="m", mode="repair")
    with patch.object(coder_agent, "_generate_sql", side_effect=RuntimeError("429 RESOURCE_EXHAUSTED")):
        run_examples(_examples()[:1], lambda ex: tiny_db, cfg, rule="spider", cache_dir=tmp_path, progress=False)
    assert load_cached(cache_path(cfg, tmp_path)) == {}


def test_cache_key_changes_with_config():
    a = RunConfig(benchmark="spider", model="m", mode="single")
    b = RunConfig(benchmark="spider", model="m", mode="repair")
    assert a.cache_key() != b.cache_key()
    assert a.cache_key() == RunConfig(benchmark="spider", model="m", mode="single").cache_key()


def test_stratified_subset_is_deterministic_and_proportional():
    exs = [Example(f"x{i}", "d", "q", "s", "easy" if i < 75 else "hard") for i in range(100)]
    sub = stratified_subset(exs, 20, seed=3)
    assert len(sub) == 20
    assert sum(e.difficulty == "easy" for e in sub) == 15
    assert [e.id for e in sub] == [e.id for e in stratified_subset(exs, 20, seed=3)]


def test_null_results_match_gold_none(tmp_path):
    # Regression: pandas turned NULL into NaN, which never equals None.
    path = tmp_path / "n.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE t (a INTEGER, b REAL); INSERT INTO t VALUES (1, NULL), (2, 3.5);")
    conn.close()
    ex = Example("n1", "n", "q", "SELECT a, b FROM t", "easy")
    cfg = RunConfig(benchmark="unit", model="m", mode="single")
    with patch.object(coder_agent, "_generate_sql", return_value="SELECT a, b FROM t"):
        [r] = run_examples([ex], lambda e: path, cfg, rule="spider", progress=False)
    assert r.match
