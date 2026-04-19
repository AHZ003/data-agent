"""Smoke test for the VCR-style LLM cassette system.

Pins two things:

1. The cassette helpers themselves — `record`, `match`, and `CassetteMiss`
   must round-trip cleanly and produce a determinstic prompt hash.

2. A **pilot integration** that stands up a fake `genai.Client` from
   the cassette and drives `storyteller_agent.generate_narrative()`
   all the way through the agent's real prompt-building code, without
   hitting the network. This is the pattern future tests should copy:
   monkey-patch the agent module's `genai.Client`, drive the public
   entrypoint, assert on the returned text.

The cassette is recorded in-process (using `record(...)`) and cleaned
up via an autouse fixture, so the test doesn't leave state behind.
"""

import json
from pathlib import Path

import pytest

from core import llm_cassette
from core.llm_cassette import CassetteMiss


# ── Helper round-trip ────────────────────────────────────────────────────

def test_record_then_match_round_trip(tmp_path, monkeypatch):
    # Redirect CASSETTE_DIR at the module level so we don't touch the
    # real tests/cassettes/ folder during the round-trip test.
    monkeypatch.setattr(llm_cassette, "CASSETTE_DIR", tmp_path)
    llm_cassette.record("demo", "hello world", "hi back")
    assert llm_cassette.match("demo", "hello world") == "hi back"


def test_match_raises_cassette_miss_with_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_cassette, "CASSETTE_DIR", tmp_path)
    with pytest.raises(CassetteMiss) as ei:
        llm_cassette.match("empty", "never-recorded prompt")
    assert "empty" in str(ei.value)
    # The error must name a sha so the dev can re-record it.
    assert any(c.isalnum() for c in str(ei.value))


def test_sha_is_stable_and_short(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_cassette, "CASSETTE_DIR", tmp_path)
    llm_cassette.record("stable", "prompt-A", "resp-A")
    llm_cassette.record("stable", "prompt-A", "resp-A-new")  # overwrite
    assert llm_cassette.match("stable", "prompt-A") == "resp-A-new"
    raw = json.loads((tmp_path / "stable.json").read_text())
    # Exactly one key for prompt-A — idempotent on sha collision.
    assert len(raw) == 1
    # Key length is 24 (_sha truncates to 24 hex chars).
    assert len(next(iter(raw.keys()))) == 24


# ── Pilot: storyteller runs against a cassette, no network ───────────────

def test_storyteller_generate_narrative_via_cassette(tmp_path, monkeypatch):
    """Drive the real storyteller entrypoint through a fake Client.

    This is the pattern: precompute the prompt the agent WILL build
    using its own helper (_build_narrative_prompt), seed a cassette
    under that exact key, patch `genai.Client`, and assert the
    narrative returned matches the cassette text. If an agent prompt
    template changes, the sha shifts and the test flags the drift.
    """
    from agents import storyteller_agent

    # Point the cassette system at a tmp dir for this test only.
    monkeypatch.setattr(llm_cassette, "CASSETTE_DIR", tmp_path)

    question = "Which region has the highest total sales?"
    sql_query = "SELECT region, SUM(sales) FROM t GROUP BY 1 ORDER BY 2 DESC"
    result_summary = "  region  sales\nCentral  1200000"
    expected_prompt = storyteller_agent._build_narrative_prompt(
        question=question,
        sql_query=sql_query,
        result_summary=result_summary,
    )
    canned = "Central region leads with $1.2M in sales."
    llm_cassette.record("storyteller_topregion", expected_prompt, canned)

    # Swap Client on the agent module so the real function routes to
    # the cassette. Note: we patch `storyteller_agent.genai.Client`,
    # not the top-level google.genai — this is the import the agent
    # actually uses at call time.
    monkeypatch.setattr(
        storyteller_agent.genai,
        "Client",
        lambda api_key=None: llm_cassette.make_replay_client("storyteller_topregion"),
    )

    out = storyteller_agent.generate_narrative(
        question=question,
        sql_query=sql_query,
        result_summary=result_summary,
    )
    assert out == canned


def test_storyteller_stream_narrative_via_cassette(tmp_path, monkeypatch):
    """The streaming code path must also replay cleanly. The fake
    client yields the full text as one chunk, which is enough to
    exercise the generator plumbing without network."""
    from agents import storyteller_agent

    monkeypatch.setattr(llm_cassette, "CASSETTE_DIR", tmp_path)

    question = "Why did profit drop?"
    expected_prompt = storyteller_agent._build_narrative_prompt(
        question=question,
        sql_query="SELECT 1",
        result_summary="row1",
    )
    llm_cassette.record("storyteller_stream", expected_prompt, "Profit dropped because X.")
    monkeypatch.setattr(
        storyteller_agent.genai,
        "Client",
        lambda api_key=None: llm_cassette.make_replay_client("storyteller_stream"),
    )

    chunks = list(storyteller_agent.stream_narrative(
        question=question,
        sql_query="SELECT 1",
        result_summary="row1",
    ))
    assert "".join(chunks) == "Profit dropped because X."
