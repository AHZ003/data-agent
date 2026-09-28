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
