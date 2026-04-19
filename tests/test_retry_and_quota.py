"""Regression tests for the bugs surfaced by the first full eval run.

See docs/postmortems/2026-04-13_eval_findings.md for the incident.

Two bugs are pinned here:

1. Coder agent used to silently retry on Gemini 429 / RESOURCE_EXHAUSTED,
   treating the error string as if it were an SQL query. This test
   mocks a 429 and asserts the coder short-circuits with a
   QuotaExhausted sentinel in a single attempt.

2. Orchestrator's `retry_count` was read but never written, so a
   persistently-rejecting critic produced an infinite retry loop.
   This test drives `should_retry` directly and asserts that after
   `coder_node` runs, the counter has incremented — plus asserts the
   quota-sentinel short-circuits the retry edge.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from agents import coder_agent, orchestrator
from agents.coder_agent import QUOTA_SENTINEL
from core.database import Database
from models.analysis_plan import (
    CodeResult,
    SemanticSchema,
    ColumnProfile,
    ColumnRole,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def toy_schema():
    return SemanticSchema(
        table_name="t",
        row_count=3,
        column_count=2,
        columns=[
            ColumnProfile(name="a", dtype="int64", role=ColumnRole.MEASURE,
                          null_count=0, null_percentage=0.0, unique_count=3),
            ColumnProfile(name="b", dtype="object", role=ColumnRole.DIMENSION,
                          null_count=0, null_percentage=0.0, unique_count=3),
        ],
    )


@pytest.fixture
def toy_db():
    d = Database()
    d.load_dataframe(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}), "t")
    return d


# ── Coder short-circuit on 429 ───────────────────────────────────────────

def test_coder_shortcircuits_on_429(toy_schema, toy_db):
    """A Gemini 429 must NOT trigger the generic retry loop."""
    call_count = {"n": 0}

    def fake_generate_sql(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError(
            "429 RESOURCE_EXHAUSTED. You exceeded your current quota"
        )

    with patch.object(coder_agent, "_generate_sql", side_effect=fake_generate_sql):
        df, result = coder_agent.execute_analysis("test q", toy_schema, toy_db)

    assert df is None
    assert result.success is False
    assert QUOTA_SENTINEL in result.error
    # CRITICAL: exactly ONE call, not MAX_RETRY_ATTEMPTS.
    assert call_count["n"] == 1, (
        f"Coder retried on quota error ({call_count['n']} calls). "
        "This is the pre-fix behavior that caused the eval to burn 2,703 "
        "coder spans in 25 minutes."
    )
    # Ensure the sql_query field is NOT polluted with the error string.
    assert "RESOURCE_EXHAUSTED" not in (result.sql_query or "")


def test_coder_preserves_last_sql_on_sql_error(toy_schema, toy_db):
    """When SQL execution fails, the final CodeResult keeps the last SQL tried."""
    sqls = ["SELECT nonexistent FROM t", "SELECT still_bad FROM t", "SELECT also_bad FROM t"]

    def fake_generate_sql(*args, **kwargs):
        return sqls.pop(0)

    with patch.object(coder_agent, "_generate_sql", side_effect=fake_generate_sql):
        df, result = coder_agent.execute_analysis("q", toy_schema, toy_db)

    assert result.success is False
    # The bug being prevented: sql_query used to be overwritten with the
    # error string, making the field useless for debugging.
    assert result.sql_query.lower().startswith("select")


def test_coder_detects_variant_quota_messages(toy_schema, toy_db):
    """Any of the known quota phrasings should short-circuit."""
    for msg in [
        "429: Too Many Requests",
        "Rate limit exceeded",
        "You have hit your daily quota",
        "RESOURCE_EXHAUSTED",
    ]:
        with patch.object(
            coder_agent,
            "_generate_sql",
            side_effect=RuntimeError(msg),
        ):
            df, result = coder_agent.execute_analysis("q", toy_schema, toy_db)
            assert result.success is False
            assert QUOTA_SENTINEL in result.error, f"missed variant: {msg!r}"


# ── Orchestrator retry counter ───────────────────────────────────────────

def test_should_retry_honors_counter():
    """should_retry must stop retrying once retry_count hits 3."""
    validated_state = {
        "validation": {"status": "validated"},
        "retry_count": 1,
        "error": None,
    }
    assert orchestrator.should_retry(validated_state) == "continue"

    rejected_state_fresh = {
        "validation": {"status": "rejected"},
        "retry_count": 1,
        "error": None,
    }
    assert orchestrator.should_retry(rejected_state_fresh) == "retry"

    rejected_state_maxed = {
        "validation": {"status": "rejected"},
        "retry_count": 3,
        "error": None,
    }
    assert orchestrator.should_retry(rejected_state_maxed) == "continue"


def test_should_retry_shortcircuits_on_quota():
    """A QuotaExhausted error must skip retries even when critic rejects."""
    state = {
        "validation": {"status": "rejected"},
        "retry_count": 1,
        "error": "QuotaExhausted: 429 RESOURCE_EXHAUSTED",
    }
    assert orchestrator.should_retry(state) == "continue"


def test_coder_node_increments_retry_count(toy_schema, toy_db):
    """coder_node must increment retry_count so the bound is effective."""
    mock_result = CodeResult(sql_query="SELECT 1", success=True,
                              row_count=1, columns=["a"])

    with patch.object(
        coder_agent,
        "execute_analysis",
        return_value=(pd.DataFrame({"a": [1]}), mock_result),
    ):
        state = {
            "question": "q",
            "schema": toy_schema.model_dump(),
            "db": toy_db,
            "retry_count": 0,
            "agent_log": [],
        }
        out = orchestrator.coder_node(state)
        assert out["retry_count"] == 1

        out2 = orchestrator.coder_node(out)
        assert out2["retry_count"] == 2

        out3 = orchestrator.coder_node(out2)
        assert out3["retry_count"] == 3


def test_retry_bound_regression_full_loop(toy_schema, toy_db):
    """End-to-end: a persistently-rejecting critic must NOT loop forever.

    Simulates the exact pathological case the eval harness caught —
    coder always succeeds at the SQL level but critic always rejects —
    and asserts that should_retry returns 'continue' after retry_count
    hits the cap. Before the fix, coder_node never wrote retry_count,
    so this would never terminate.
    """
    mock_result = CodeResult(sql_query="SELECT 1", success=True,
                              row_count=1, columns=["a"])

    state = {
        "question": "q",
        "schema": toy_schema.model_dump(),
        "db": toy_db,
        "retry_count": 0,
        "agent_log": [],
        "validation": {"status": "rejected"},  # critic always rejects
        "error": None,
    }
    with patch.object(
        coder_agent,
        "execute_analysis",
        return_value=(pd.DataFrame({"a": [1]}), mock_result),
    ):
        iterations = 0
        while orchestrator.should_retry(state) == "retry":
            state = orchestrator.coder_node(state)
            state["validation"] = {"status": "rejected"}
            iterations += 1
            assert iterations < 10, "retry loop didn't terminate — bug reintroduced"

    assert state["retry_count"] == 3
