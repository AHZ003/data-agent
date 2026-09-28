"""Semantic cache and the cost-confirmation interrupt node."""

from unittest.mock import patch

import pandas as pd

from core.datasource import Column, DataSource, Table
from core.semantic_cache import SemanticCache, schema_version
from retrieval.embed import HashingEmbedder


def _ds(cols=("a", "b")):
    return DataSource("d", [Table("t", [Column(c, "INTEGER") for c in cols])])


def test_schema_version_ignores_data_but_not_schema():
    a = _ds()
    b = _ds()
    b.tables[0].row_count = 999
    b.tables[0].columns[0].sample_values = [1, 2]
    assert schema_version(a) == schema_version(b)
    assert schema_version(a) != schema_version(_ds(("a", "c")))


def test_hit_on_paraphrase_miss_on_other_question_and_other_schema():
    cache = SemanticCache(HashingEmbedder(), threshold=0.9)
    v = schema_version(_ds())
    cache.store("What is the total revenue?", v, "SELECT SUM(x) FROM t", cost_usd=0.001)
    assert cache.lookup("what is the total revenue", v) == "SELECT SUM(x) FROM t"
    assert cache.lookup("How many customers churned last year?", v) is None
    assert cache.lookup("What is the total revenue?", "other-version") is None
    assert cache.stats.hits == 1 and cache.stats.misses == 2
    assert cache.stats.usd_saved == 0.001


def test_lru_bound():
    cache = SemanticCache(HashingEmbedder(), max_entries=2)
    for i in range(3):
        cache.store(f"question number {i}", "v", f"SELECT {i}")
    assert sum(len(e) for e in cache._by_version.values()) == 2


def test_coder_node_uses_cache_and_skips_llm(monkeypatch):
    from agents import coder_agent, orchestrator
    from core.database import Database
    from models.analysis_plan import SemanticSchema

    db = Database()
    db.load_dataframe(pd.DataFrame({"x": [1, 2, 3]}), "t")
    cache = SemanticCache(HashingEmbedder())
    monkeypatch.setattr(orchestrator, "default_cache", lambda: cache)
    schema = SemanticSchema(table_name="t", row_count=3, column_count=1, columns=[])
    state = lambda: {"question": "total of x", "schema": schema.model_dump(), "db": db,  # noqa: E731
                     "retry_count": 0, "agent_log": [], "history": []}
    with patch.object(coder_agent, "_generate_sql", return_value='SELECT SUM("x") AS s FROM "t"') as gen:
        first = orchestrator.coder_node(state())
        second = orchestrator.coder_node(state())
    assert gen.call_count == 1
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert int(second["result_df"]["s"][0]) == 6


def test_confirm_cost_node_runs_or_cancels(monkeypatch):
    from agents import orchestrator

    class FakeBQ:
        def execute_query(self, sql, confirm_cost=False):
            return (pd.DataFrame({"n": [1]}), None) if confirm_cost else (None, "CostConfirmationRequired")

    base = {"db": FakeBQ(), "sql_query": "SELECT 1", "error": "CostConfirmationRequired: 12 GiB"}
    monkeypatch.setattr(orchestrator, "interrupt", lambda payload: True)
    out = orchestrator.confirm_cost_node(dict(base))
    assert out["error"] is None and out["result_df"]["n"][0] == 1
    monkeypatch.setattr(orchestrator, "interrupt", lambda payload: False)
    out = orchestrator.confirm_cost_node(dict(base))
    assert out["error"].startswith("Cancelled")
