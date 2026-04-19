"""Unit tests for chaos/robustness scorers in benchmarks/scorers.py.

The chaos cases in cases.yaml depend on these scorers correctly
rewarding *graceful refusal* and penalizing *hallucinated confidence*.
Without these tests a bad scorer edit could flip the polarity —
silently making the harness reward the thing it was designed to catch.
"""

import pandas as pd

from benchmarks.scorers import (
    score_expect_empty,
    score_narrative_must_not_mention,
    score_expect_warning,
    score_case,
)


# ── score_expect_empty ───────────────────────────────────────────────────

def test_expect_empty_noop_when_not_required():
    r = score_expect_empty(pd.DataFrame({"a": [1, 2, 3]}), expect_empty=False)
    assert r.passed is True
    assert r.weight == 0.0  # no-op


def test_expect_empty_pass_on_empty_df():
    r = score_expect_empty(pd.DataFrame(), expect_empty=True)
    assert r.passed and r.score == 1.0


def test_expect_empty_fail_on_nonempty_df():
    r = score_expect_empty(pd.DataFrame({"a": [1]}), expect_empty=True)
    assert not r.passed and r.score == 0.0


def test_expect_empty_pass_on_none():
    r = score_expect_empty(None, expect_empty=True)
    assert r.passed


# ── score_narrative_must_not_mention ─────────────────────────────────────

def test_narrative_clean_passes():
    r = score_narrative_must_not_mention(
        "The result is inconclusive.", banned=["Mars", "quantum_flux"]
    )
    assert r.passed and r.score == 1.0


def test_narrative_hallucinates_one_fails():
    r = score_narrative_must_not_mention(
        "Sales on Mars totaled $1.2M.", banned=["Mars", "quantum_flux"]
    )
    assert not r.passed and r.score == 0.0
    assert "Mars" in r.reason


def test_narrative_case_insensitive():
    r = score_narrative_must_not_mention(
        "QUANTUM_FLUX averaged 0.5 across regions.", banned=["quantum_flux"]
    )
    assert not r.passed


def test_narrative_missing_is_ok():
    r = score_narrative_must_not_mention(None, banned=["Mars"])
    assert r.passed  # no narrative means no hallucination


# ── score_expect_warning ─────────────────────────────────────────────────

def test_expect_warning_pass_on_nonempty():
    r = score_expect_warning({"warnings": ["n<30 sample"]}, expect=True)
    assert r.passed


def test_expect_warning_fail_on_empty():
    r = score_expect_warning({"warnings": []}, expect=True)
    assert not r.passed


def test_expect_warning_fail_on_missing():
    r = score_expect_warning(None, expect=True)
    assert not r.passed


# ── end-to-end: chaos case score_case dispatch ───────────────────────────

def test_score_case_dispatches_chaos_scorers():
    """A chaos case with all three new expected fields should produce
    all three new ScoreResults in the card. The bug this catches:
    forgetting to wire a new scorer into `score_case`."""
    case = {
        "id": "chaos_test",
        "tags": ["chaos"],
        "question": "Will it rain tomorrow?",
        "expected": {
            "expect_empty_result": True,
            "narrative_must_not_mention": ["tomorrow"],
            "expect_warnings_nonempty": True,
        },
    }
    # Perfect graceful-refusal output
    agent_output = {
        "sql_query": "",
        "result_df": pd.DataFrame(),
        "narrative": "The dataset cannot answer this.",
        "validation": {"warnings": ["off-topic"]},
        "chart_config": None,
    }
    card = score_case(case, agent_output, duration=0.1, error=None, use_llm_judge=False)
    names = [s.name for s in card.scores]
    assert "expect_empty" in names
    assert "narrative_no_hallucinate" in names
    assert "expect_warning" in names
    assert card.passed is True


def test_score_case_chaos_penalizes_hallucination():
    case = {
        "id": "chaos_hallucinate",
        "tags": ["chaos"],
        "question": "What is the capital of France?",
        "expected": {
            "narrative_must_not_mention": ["Paris", "France"],
        },
    }
    agent_output = {
        "sql_query": "SELECT 1",
        "result_df": pd.DataFrame({"x": [1]}),
        "narrative": "The capital of France is Paris.",
        "validation": {"warnings": [], "confidence": 90},
        "chart_config": None,
    }
    card = score_case(case, agent_output, duration=0.1, error=None, use_llm_judge=False)
    assert not card.passed
    assert any(s.name == "narrative_no_hallucinate" and not s.passed for s in card.scores)


def test_score_case_chaos_execution_weight_relaxed():
    """A chaos case that errors out should use weight=1.0 for
    execution, not 2.0 — graceful refusal isn't a prod-crash."""
    case = {"id": "chaos_err", "tags": ["chaos"], "question": "x", "expected": {}}
    card = score_case(case, {}, duration=0.1, error="SQLGuardError: blocked", use_llm_judge=False)
    exec_scores = [s for s in card.scores if s.name == "execution"]
    assert len(exec_scores) == 1
    assert exec_scores[0].weight == 1.0
