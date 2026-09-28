"""Tests for the structured tracing module."""

import json
import os
from pathlib import Path

import pytest

from core import tracing


@pytest.fixture
def trace_path(tmp_path, monkeypatch):
    p = tmp_path / "trace.jsonl"
    monkeypatch.setenv("DATAAGENT_TRACE_PATH", str(p))
    return p


def test_run_and_span_round_trip(trace_path):
    rid = tracing.new_run("what are top sales?", dataset="superstore")
    with tracing.span("coder", model="gemini-2.5-flash") as s:
        s.set_tokens(in_=1000, out_=200)
        s.add_metadata(sql="SELECT 1")
    tracing.end_run()

    lines = [json.loads(l) for l in trace_path.read_text().splitlines()]
    kinds = [l.get("kind") for l in lines]
    assert "run_start" in kinds
    assert "run_end" in kinds

    span_records = [l for l in lines if l.get("span_id")]
    assert len(span_records) == 1
    sp = span_records[0]
    assert sp["agent_name"] == "coder"
    assert sp["run_id"] == rid
    assert sp["tokens_in"] == 1000
    assert sp["tokens_out"] == 200
    assert sp["cost_usd"] > 0
    assert sp["status"] == "ok"
    assert sp["metadata"]["sql"] == "SELECT 1"


def test_span_captures_errors(trace_path):
    tracing.new_run("bad question")
    with pytest.raises(RuntimeError):
        with tracing.span("coder"):
            raise RuntimeError("boom")
    tracing.end_run(status="error", error="boom")

    spans = [
        json.loads(l) for l in trace_path.read_text().splitlines()
        if "span_id" in json.loads(l)
    ]
    assert len(spans) == 1
    assert spans[0]["status"] == "error"
    assert "boom" in spans[0]["error"]


def test_run_summary_aggregates(trace_path):
    rid = tracing.new_run("q")
    with tracing.span("planner", model="gemini-2.5-flash") as s:
        s.set_tokens(in_=500, out_=100)
    with tracing.span("coder", model="gemini-2.5-flash") as s:
        s.set_tokens(in_=800, out_=300)
    tracing.end_run()

    summary = tracing.run_summary(rid, path=trace_path)
    assert summary["run_id"] == rid
    assert summary["n_spans"] == 2
    assert summary["total_tokens_in"] == 1300
    assert summary["total_tokens_out"] == 400
    assert summary["total_cost_usd"] > 0
    assert summary["errors"] == 0
    assert summary["agents"] == ["planner", "coder"]


def test_cost_differs_by_model(trace_path):
    tracing.new_run("q")
    with tracing.span("a", model="gemini-2.5-flash") as s:
        s.set_tokens(in_=1_000_000, out_=0)
    with tracing.span("b", model="gemini-2.5-pro") as s:
        s.set_tokens(in_=1_000_000, out_=0)
    tracing.end_run()

    lines = [json.loads(l) for l in trace_path.read_text().splitlines() if "span_id" in json.loads(l)]
    flash = next(l for l in lines if l["agent_name"] == "a")
    pro = next(l for l in lines if l["agent_name"] == "b")
    assert pro["cost_usd"] > flash["cost_usd"]


# ── LLM usage accounting ─────────────────────────────────────────────────

class _UM:
    def __init__(self, tin, tout, thoughts=0):
        self.prompt_token_count = tin
        self.candidates_token_count = tout
        self.thoughts_token_count = thoughts


class _Resp:
    def __init__(self, *args):
        self.usage_metadata = _UM(*args)


def test_record_usage_goes_to_innermost_span(trace_path):
    tracing.new_run("q")
    with tracing.span("outer", model="gemini-2.5-flash"):
        with tracing.span("coder", model="gemini-2.5-flash"):
            tracing.record_usage(_Resp(100, 20, 5), "gemini-2.5-flash")
    tracing.end_run()
    spans = {s["agent_name"]: s for s in tracing.read_spans(trace_path) if "span_id" in s}
    assert spans["coder"]["tokens_in"] == 100
    assert spans["coder"]["tokens_out"] == 25  # thinking tokens billed as output
    assert spans["outer"]["tokens_in"] == 0


def test_usage_scope_accumulates_calls_and_cost(trace_path):
    with tracing.usage_scope() as u:
        tracing.record_usage(_Resp(1_000_000, 0), "gemini-2.5-flash")
        tracing.record_usage(_Resp(0, 1_000_000), "gemini-2.5-flash")
    assert u.calls == 2
    assert u.tokens_in == 1_000_000 and u.tokens_out == 1_000_000
    assert u.cost_usd == pytest.approx(0.15 + 0.60)
    # Closed scopes stop collecting.
    tracing.record_usage(_Resp(5, 5), "gemini-2.5-flash")
    assert u.calls == 2


def test_record_usage_ignores_responses_without_metadata(trace_path):
    with tracing.usage_scope() as u:
        tracing.record_usage(object(), "gemini-2.5-flash")
    assert u.calls == 0
