"""Retrieval components: tokenization, BM25, RRF, schema linking, values."""

import sqlite3

import pytest

from core.datasource import DataSource
from retrieval.bm25 import BM25
from retrieval.embed import HashingEmbedder
from retrieval.fusion import rrf
from retrieval.gold_schema import gold_items
from retrieval.schema_linking import SchemaLinker, prune
from retrieval.text import split_identifier, tokenize
from retrieval.values import build


def test_split_identifier_handles_camel_snake_and_dots():
    assert split_identifier("BillingCountry") == ["billing", "country"]
    assert split_identifier("team_long_name") == ["team", "long", "name"]
    assert split_identifier("T2.buildUpPlayPassingClass") == ["t2", "build", "up", "play", "passing", "class"]
    assert split_identifier("HTTPServer") == ["http", "server"]


def test_tokenize_stems_plurals_and_drops_stopwords():
    assert tokenize("How many customers are there in the cities?") == ["customer", "city"]


def test_bm25_prefers_rarer_matching_terms():
    bm = BM25([["customer", "name"], ["invoice", "total"], ["customer", "country"]])
    s = bm.scores(["customer", "country"])
    assert s[2] > s[0] > s[1] == 0


def test_rrf_rewards_agreement():
    assert rrf([["a", "b", "c"], ["b", "a", "c"]])[:2] in (["a", "b"], ["b", "a"])
    assert rrf([["a", "b"], ["b", "c"]])[0] == "b"


def test_hashing_embedder_is_normalized_and_similar_for_near_strings():
    e = HashingEmbedder()
    v = e.embed(["billing country", "BillingCountry", "track milliseconds"])
    assert v.shape == (3, 512)
    assert abs((v[0] ** 2).sum() - 1) < 1e-9
    assert v[0] @ v[1] > v[0] @ v[2]


@pytest.fixture
def chinook():
    from core.demo import SAMPLE_DB
    return SAMPLE_DB, DataSource.from_sqlite(str(SAMPLE_DB))


def test_schema_linking_ranks_relevant_columns_first(chinook):
    _, ds = chinook
    ranked = SchemaLinker(ds).rank("total invoice amount billed to each billing country")
    assert ("Invoice", "BillingCountry") in ranked[:3]
    assert ("Invoice", "Total") in ranked[:5]


def test_prune_keeps_join_keys_and_bridge_tables(chinook):
    _, ds = chinook
    pruned = prune(ds, {("Playlist", "Name"), ("Track", "Name")})
    names = [t.name for t in pruned.tables]
    assert "PlaylistTrack" in names  # bridge between Playlist and Track
    pt = pruned.table("PlaylistTrack")
    assert {c.name for c in pt.columns} == {"PlaylistId", "TrackId"}
    track = pruned.table("Track")
    assert {"TrackId", "Name"} <= {c.name for c in track.columns}
    assert "Composer" not in {c.name for c in track.columns}
    assert all(fk.ref_table in names for t in pruned.tables for fk in t.foreign_keys)


def test_small_schema_is_not_pruned():
    ds = DataSource.from_dataframes({"t": __import__("pandas").DataFrame({"a": [1], "b": ["x"]})})
    assert SchemaLinker(ds).link("anything", k=1) is ds


def test_gold_items_resolves_aliases_per_scope(chinook):
    _, ds = chinook
    tables, cols = gold_items(
        "SELECT T1.Name FROM Artist AS T1 JOIN Album AS T2 ON T1.ArtistId = T2.ArtistId "
        "WHERE T2.AlbumId IN (SELECT T1.AlbumId FROM Track AS T1 WHERE Milliseconds > 1)", ds)
    assert tables == {"artist", "album", "track"}
    assert ("track", "milliseconds") in cols and ("artist", "name") in cols
    assert ("track", "albumid") in cols and ("album", "albumid") in cols


def test_value_index_finds_fuzzy_and_exact_matches(chinook):
    path, ds = chinook
    idx = build(sqlite3.connect(path), ds)
    hits = idx.match("How many invoices were billed to germany?")
    assert any(m.column == "BillingCountry" and m.value == "Germany" for m in hits)
    fuzzy = idx.match("tracks by Led Zepelin")  # misspelled
    assert any(m.value == "Led Zeppelin" for m in fuzzy)
    assert not any(m.span.lower() == "what" for m in idx.match("What is the total?"))


# ── Query memory ─────────────────────────────────────────────────────────

from retrieval.memory import LeakageError, MemoryItem, QueryMemory  # noqa: E402


def test_memory_refuses_evaluation_examples():
    mem = QueryMemory()
    for bad in ("spider-dev-12", "bird-minidev-3", "chk_001"):
        with pytest.raises(LeakageError):
            mem.add(MemoryItem(bad, "q", "SELECT 1", "d", "x"))


def test_memory_search_ranks_similar_and_excludes_same_db():
    mem = QueryMemory(HashingEmbedder())
    mem.add(MemoryItem("a", "How many singers are there?", "SELECT count(*) FROM singer", "concert", "t"))
    mem.add(MemoryItem("b", "List the names of all stadiums", "SELECT name FROM stadium", "concert", "t"))
    mem.add(MemoryItem("c", "How many singers do we have?", "SELECT count(*) FROM singer", "music", "t"))
    top = mem.search("How many singers exist?", k=2)
    assert {top[0].id, top[1].id} == {"a", "c"}
    assert [i.id for i in mem.search("How many singers exist?", k=3, exclude_db="concert")] == ["c"]


def test_memory_roundtrip(tmp_path):
    mem = QueryMemory()
    mem.add(MemoryItem("a", "q", "SELECT 1", "d", "t"))
    mem.save(tmp_path / "m.jsonl")
    assert QueryMemory.load(tmp_path / "m.jsonl").items == mem.items


# ── Semantic layer ───────────────────────────────────────────────────────

from semantic_layer import SemanticLayer  # noqa: E402


def test_chinook_semantic_layer_is_valid(chinook):
    _, ds = chinook
    layer = SemanticLayer.for_datasource("chinook")
    assert layer.validate(ds) == []
    assert layer.relevant_metrics("Which artists earned the most?")[0].name == "revenue"
    annotated = layer.annotate(ds)
    assert "not what customers paid" in annotated.table("Track").column("UnitPrice").description


def test_semantic_layer_validate_catches_unknown_columns(chinook, tmp_path):
    _, ds = chinook
    p = tmp_path / "bad.yaml"
    p.write_text('datasource: x\nmetrics:\n  - {name: m, description: d, sql: \'SUM("Invoice"."Nope")\'}\n'
                 "columns:\n  Invoice.Missing: d\n")
    problems = SemanticLayer.load(p).validate(ds)
    assert len(problems) == 2


# ── Retriever through the real coder path ────────────────────────────────

def test_retriever_context_reaches_the_prompt(chinook):
    from unittest.mock import patch

    from agents import coder_agent
    from core.database import Database
    from retrieval.pipeline import RetrievalConfig, Retriever

    path, _ = chinook
    db = Database.from_sqlite(str(path), name="chinook")
    mem = QueryMemory()
    mem.add(MemoryItem("x1", "Total sales per country?", "SELECT country, SUM(amount) FROM sales GROUP BY 1",
                       "other", "t"))
    r = Retriever(RetrievalConfig(schema_k=10, values=True, memory_k=1, semantic=True, embedder="hashing"),
                  memory=mem)
    prompts = []

    def fake(q, ds, error_context="", model=None, context=None):
        prompts.append(coder_agent._build_sql_prompt(q, ds, context))
        return 'SELECT SUM("Total") FROM "Invoice" WHERE "BillingCountry" = \'Germany\''

    with patch.object(coder_agent, "_generate_sql", side_effect=fake):
        df, res = coder_agent.generate_and_execute("How much revenue did we earn from germany?", db, retriever=r)
    assert res.success
    p = prompts[0]
    assert "\"Invoice\".\"BillingCountry\" contains 'Germany'" in p
    assert "revenue: Money received" in p
    assert "Q: Total sales per country?" in p
    assert 'Table "Playlist"' not in p  # pruned away


def test_retrieval_config_from_flags():
    from retrieval.pipeline import RetrievalConfig
    assert RetrievalConfig.from_flags({}) is None
    assert RetrievalConfig.from_flags({"values": True, "unrelated": 1}).values is True
