"""Bounded backoff for rate limits in batch runs."""

import pytest

from core.llm import with_rate_limit_backoff


def _is_429(e):
    return "429" in str(e)


def test_no_retries_by_default_raises_immediately(monkeypatch):
    monkeypatch.delenv("DATAAGENT_LLM_BACKOFF_RETRIES", raising=False)
    calls = []

    def call():
        calls.append(1)
        raise RuntimeError("429 Too Many Requests")

    with pytest.raises(RuntimeError):
        with_rate_limit_backoff(call, _is_429, sleep=lambda s: None)
    assert len(calls) == 1


def test_retries_rate_limits_then_succeeds():
    outcomes = [RuntimeError("429"), RuntimeError("429"), "ok"]
    sleeps = []

    def call():
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o

    assert with_rate_limit_backoff(call, _is_429, retries=3, sleep=sleeps.append) == "ok"
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0] * 0.5  # grows (with jitter)


def test_gives_up_after_bounded_retries():
    calls = []

    def call():
        calls.append(1)
        raise RuntimeError("429")

    with pytest.raises(RuntimeError):
        with_rate_limit_backoff(call, _is_429, retries=2, sleep=lambda s: None)
    assert len(calls) == 3


def test_non_rate_limit_errors_are_not_retried():
    calls = []

    def call():
        calls.append(1)
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        with_rate_limit_backoff(call, _is_429, retries=5, sleep=lambda s: None)
    assert len(calls) == 1
