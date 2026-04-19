"""Structured JSON tracing for the agent crew.

Every agent node emits a span via `emit_span(...)`. Spans carry:
  - run_id      : correlation id for all spans in one user question
  - agent_name  : planner | coder | critic | ...
  - duration_ms : wall-clock
  - tokens_in/out : best-effort token accounting
  - cost_usd    : derived from model pricing table
  - status      : ok | error
  - metadata    : arbitrary JSON (SQL excerpt, chart type, confidence, etc.)

Writes newline-delimited JSON to `outputs/traces/<date>.jsonl` by default
(override with DATAAGENT_TRACE_PATH) so a downstream viewer can slice by
run_id, agent, or model. This is the minimum wire for observability that
replaces "I read the logs" with "I query my trace store."
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# Gemini 2.0 / 2.5 Flash pricing as of 2026. Adjust when Google changes
# the rate card. Values are USD per 1M tokens.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash":      {"in": 0.10, "out": 0.40},
    "gemini-2.0-flash-exp":  {"in": 0.10, "out": 0.40},
    "gemini-2.5-flash":      {"in": 0.15, "out": 0.60},
    "gemini-2.5-pro":        {"in": 1.25, "out": 5.00},
    "default":               {"in": 0.15, "out": 0.60},
}


def _trace_path() -> Path:
    override = os.environ.get("DATAAGENT_TRACE_PATH")
    if override:
        return Path(override)
    root = Path(__file__).resolve().parent.parent
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return root / "outputs" / "traces" / f"{day}.jsonl"


def _cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    rates = MODEL_PRICING.get(model) or MODEL_PRICING["default"]
    return round(
        (tokens_in / 1_000_000) * rates["in"] + (tokens_out / 1_000_000) * rates["out"],
        6,
    )


@dataclass
class Span:
    run_id: str
    span_id: str
    parent_id: Optional[str]
    agent_name: str
    model: Optional[str]
    started_at: str
    duration_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    status: str                       # "ok" | "error"
    error: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


# ── Run-level context (thread-local for Streamlit compat) ────────────────

_ctx = threading.local()


def new_run(question: str, dataset: Optional[str] = None) -> str:
    """Start a new correlated run. Returns the run_id."""
    rid = uuid.uuid4().hex[:16]
    _ctx.run_id = rid
    _ctx.parent_id = None
    emit_event(
        kind="run_start",
        agent_name="orchestrator",
        metadata={"question": question[:500], "dataset": dataset},
    )
    return rid


def end_run(status: str = "ok", error: Optional[str] = None) -> None:
    emit_event(
        kind="run_end",
        agent_name="orchestrator",
        metadata={"status": status, "error": error},
    )
    _ctx.run_id = None
    _ctx.parent_id = None


def current_run_id() -> str:
    rid = getattr(_ctx, "run_id", None)
    if not rid:
        rid = uuid.uuid4().hex[:16]
        _ctx.run_id = rid
    return rid


# ── Emitters ─────────────────────────────────────────────────────────────

def _write(span_dict: dict) -> None:
    path = _trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(span_dict, default=str) + "\n")


def emit_event(
    kind: str,
    agent_name: str,
    metadata: Optional[dict] = None,
) -> None:
    """Emit a zero-duration marker event (e.g. run_start, run_end)."""
    _write(
        {
            "kind": kind,
            "run_id": current_run_id(),
            "agent_name": agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
    )


@contextmanager
def span(
    agent_name: str,
    model: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """Context manager wrapping an agent step. Auto-emits a Span on exit.

    Usage:
        with tracing.span("coder", model="gemini-2.5-flash") as s:
            result = do_work()
            s.set_tokens(in_=123, out_=456)
            s.add_metadata(sql=result.sql[:200])
    """
    sid = uuid.uuid4().hex[:12]
    parent = getattr(_ctx, "parent_id", None)
    _ctx.parent_id = sid
    started = time.time()
    tokens_in = 0
    tokens_out = 0
    meta: dict = dict(metadata or {})
    status = "ok"
    error: Optional[str] = None

    class _Handle:
        def set_tokens(self, in_: int = 0, out_: int = 0) -> None:
            nonlocal tokens_in, tokens_out
            tokens_in += int(in_ or 0)
            tokens_out += int(out_ or 0)

        def add_metadata(self, **kwargs: Any) -> None:
            meta.update(kwargs)

    handle = _Handle()
    try:
        yield handle
    except Exception as e:
        status = "error"
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        duration_ms = round((time.time() - started) * 1000, 2)
        span_obj = Span(
            run_id=current_run_id(),
            span_id=sid,
            parent_id=parent,
            agent_name=agent_name,
            model=model,
            started_at=datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
            duration_ms=duration_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=_cost_usd(model or "default", tokens_in, tokens_out),
            status=status,
            error=error,
            metadata=meta,
        )
        _write(asdict(span_obj))
        _ctx.parent_id = parent


# ── Query API (for a future trace-viewer UI) ─────────────────────────────

def read_spans(path: Optional[Path] = None, run_id: Optional[str] = None) -> list[dict]:
    p = path or _trace_path()
    if not p.exists():
        return []
    out: list[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run_id and rec.get("run_id") != run_id:
                continue
            out.append(rec)
    return out


def run_summary(run_id: str, path: Optional[Path] = None) -> dict:
    """Aggregate a single run into headline metrics."""
    spans = read_spans(path=path, run_id=run_id)
    agent_spans = [s for s in spans if "span_id" in s]
    total_ms = sum(s.get("duration_ms", 0) for s in agent_spans)
    total_in = sum(s.get("tokens_in", 0) for s in agent_spans)
    total_out = sum(s.get("tokens_out", 0) for s in agent_spans)
    total_cost = sum(s.get("cost_usd", 0) for s in agent_spans)
    errors = [s for s in agent_spans if s.get("status") == "error"]
    return {
        "run_id": run_id,
        "n_spans": len(agent_spans),
        "total_duration_ms": round(total_ms, 2),
        "total_tokens_in": total_in,
        "total_tokens_out": total_out,
        "total_cost_usd": round(total_cost, 6),
        "errors": len(errors),
        "agents": [s.get("agent_name") for s in agent_spans],
    }
