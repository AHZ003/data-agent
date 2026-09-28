"""DataAgent HTTP API.

    uvicorn api.main:app --reload            # local
    open http://localhost:8000/docs          # OpenAPI

Endpoints
---------
POST /v1/datasources            upload CSV/Parquet files -> a DuckDB workspace
GET  /v1/datasources            list workspaces (the Chinook sample is preloaded)
POST /v1/analyze                ask a question; Server-Sent Events stream
POST /v1/runs/{id}/resume       answer a clarifying question / approve a cost
GET  /v1/runs/{id}              final result of a run
POST /v1/runs/{id}/sql          re-run user-edited SQL (guarded; edit is logged)
GET  /v1/metrics                cache hit rate, $ saved, run counts

SSE events, in order: plan, sql, result_preview, chart_spec,
narrative_token*, done — or clarify / confirm_cost (the run pauses; resume
it) or error. Conversations are LangGraph threads persisted by a
checkpointer (SQLite locally, Postgres when DATAAGENT_CHECKPOINT_DSN is
set), so a follow-up with the same conversation_id sees earlier turns.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents import orchestrator, storyteller_agent
from api.auth import RateLimiter, require_key
from core import audit, faithfulness, pii, resources, tracing
from core.demo import SAMPLE_QUESTIONS, load_sample_workspace
from core.llm import api_key_override

ROOT = Path(__file__).resolve().parent.parent
EDITS_LOG = ROOT / "outputs" / "sql_edits.jsonl"
MAX_UPLOAD_MB = int(os.getenv("API_MAX_UPLOAD_MB", "100"))


# ── Checkpointer ────────────────────────────────────────────────────────

def make_checkpointer():
    """Postgres when DATAAGENT_CHECKPOINT_DSN is set, else a local SQLite file.

    State holds DataFrames and Plotly figures, so the serializer falls back
    to pickle for types msgpack can't encode. Checkpoints are written only
    by this service, never from user input.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    serde = JsonPlusSerializer(pickle_fallback=True)
    dsn = os.getenv("DATAAGENT_CHECKPOINT_DSN")
    if dsn:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(dsn, kwargs={"autocommit": True, "prepare_threshold": 0}, open=True)
        saver = PostgresSaver(pool, serde=serde)
        saver.setup()
        return saver
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = os.getenv("DATAAGENT_CHECKPOINT_PATH", str(ROOT / "outputs" / "checkpoints.sqlite"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False), serde=serde)


# ── In-process registries ───────────────────────────────────────────────

class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.datasources: dict[str, dict] = {}   # id -> {name, engine, tables}
        self.runs: dict[str, dict] = {}          # run_id -> {conversation_id, status, result}
        self.graph = None
        self.limiter = RateLimiter()
        self.counters = {"runs": 0, "interrupted": 0, "errors": 0}


S = _State()


def _register_sample() -> None:
    db = load_sample_workspace()
    resources.register(db, None, datasource_id="sample")
    S.datasources["sample"] = {"name": "Chinook (sample)", "engine": "sqlite",
                               "tables": db.datasource().table_names, "suggested_questions": SAMPLE_QUESTIONS}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    S.graph = orchestrator.build_graph(interactive=True).compile(checkpointer=make_checkpointer())
    _register_sample()
    yield


app = FastAPI(title="DataAgent API", version="1.0", lifespan=lifespan)


# ── Models ──────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    datasource_id: str = "sample"
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[str] = None


class ResumeRequest(BaseModel):
    answer: str | bool


class SQLEdit(BaseModel):
    sql: str = Field(min_length=1, max_length=20000)


# ── SSE helpers ─────────────────────────────────────────────────────────

def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _preview(df: Optional[pd.DataFrame], n: int = 50) -> Optional[dict]:
    if df is None:
        return None
    head = df.head(n).astype(object).where(df.head(n).notna(), None)
    return {"columns": list(df.columns), "rows": head.values.tolist(), "row_count": len(df)}


def _events_for(node: str, update: dict) -> Iterator[str]:
    if node == "planner" and update.get("plan"):
        yield sse("plan", update["plan"])
    elif node in ("coder", "confirm_cost"):
        if update.get("sql_query"):
            yield sse("sql", {"sql": update["sql_query"], "cache_hit": update.get("cache_hit")})
        if update.get("result_df") is not None:
            yield sse("result_preview", _preview(update["result_df"]))
    elif node == "visualizer" and update.get("chart") is not None:
        yield sse("chart_spec", json.loads(update["chart"].to_json()))
    elif node == "critic" and update.get("validation"):
        yield sse("validation", update["validation"])


def _run_stream(run_id: str, graph_input, gemini_key: Optional[str]) -> Iterator[str]:
    """Drive the graph for one run, translating node updates into SSE."""
    run = S.runs[run_id]
    config = {"configurable": {"thread_id": run["conversation_id"]}}
    with api_key_override(gemini_key):
        tracing.new_run(run["question"], dataset=run["datasource_id"])
        try:
            for chunk in S.graph.stream(graph_input, config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        payload = update[0].value
                        run["status"] = "interrupted"
                        S.counters["interrupted"] += 1
                        yield sse(payload["type"], {**payload, "run_id": run_id})
                        return
                    if isinstance(update, dict):
                        yield from _events_for(node, update)

            final = S.graph.get_state(config).values
            engine, _ = resources.get(run["datasource_id"])
            # Stream the narrative token by token (the graph skipped it).
            parts = []
            for token in storyteller_agent.stream_narrative(
                question=final["question"], sql_query=final.get("sql_query") or "",
                result_summary=storyteller_agent.summarize_result(
                    final.get("result_df"), pii.detect(engine.datasource())),
                chart_description=(final.get("chart_config") or {}).get("chart_type", ""),
                validation_warnings=(final.get("validation") or {}).get("warnings", []),
                prediction_info=final.get("prediction"),
            ):
                parts.append(token)
                yield sse("narrative_token", {"text": token})
            narrative = "".join(parts)
            warning = faithfulness.check(narrative, final.get("result_df"), final["question"])
            if warning:
                yield sse("faithfulness", {"warning": warning})
            run.update(status="done", result={
                "question": final.get("display_question"), "sql": final.get("sql_query"),
                "error": final.get("error"), "result": _preview(final.get("result_df")),
                "validation": final.get("validation"), "narrative": narrative,
                "faithfulness_warning": warning,
            })
            audit.record(run_id=run_id, api_key=run["key"], question=run["question"],
                         sql=final.get("sql_query"), engine=engine,
                         status="error" if final.get("error") else "ok")
            yield sse("done", {"run_id": run_id, "conversation_id": run["conversation_id"],
                               "error": final.get("error")})
            tracing.end_run(status="ok")
        except Exception as e:
            run["status"] = "error"
            S.counters["errors"] += 1
            tracing.end_run(status="error", error=str(e))
            yield sse("error", {"run_id": run_id, "message": f"{type(e).__name__}: {e}"})


_DONE = object()


async def _threaded(events: Iterator[str]):
    """Run an event generator to completion on one dedicated thread.

    Two constraints: (1) tracing is thread-local and the per-request API key
    is a ContextVar, so the whole run must stay on one thread; (2) waiting
    streams must not hold server threads. A sync generator handed to
    StreamingResponse is pulled through Starlette's 40-thread pool, and each
    open stream pinned a pool thread for its full run, which capped the
    service at ~23 req/s (see docs/performance.md). The worker thread pushes
    events onto an asyncio queue; the response awaits them without a thread.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def work():
        try:
            for e in events:
                loop.call_soon_threadsafe(q.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _DONE)

    threading.Thread(target=work, daemon=True).start()
    while (item := await q.get()) is not _DONE:
        yield item


def _limit(key_name: str) -> None:
    if not S.limiter.allow(key_name):
        raise HTTPException(status_code=429, detail="rate limit exceeded; slow down")


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/v1/datasources")
def list_datasources(_: str = Depends(require_key)):
    return [{"id": k, **{f: v[f] for f in ("name", "engine", "tables")}} for k, v in S.datasources.items()]


@app.post("/v1/datasources", status_code=201)
def upload_datasource(files: list[UploadFile] = File(...), name: str = "workspace",
                      _: str = Depends(require_key)):
    from core.duckdb_engine import DuckDBEngine

    eng = DuckDBEngine(name=name)
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            suffix = Path(f.filename or "").suffix.lower()
            if suffix not in (".csv", ".parquet"):
                raise HTTPException(400, f"{f.filename}: only .csv and .parquet are accepted")
            data = f.file.read(MAX_UPLOAD_MB * 2**20 + 1)
            if len(data) > MAX_UPLOAD_MB * 2**20:
                raise HTTPException(413, f"{f.filename}: larger than {MAX_UPLOAD_MB} MB")
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(data)
            table = "".join(ch if ch.isalnum() else "_" for ch in Path(f.filename).stem).strip("_").lower() or "data"
            eng.load_file(str(path), table)
    eng.lock()
    df = eng.conn.execute(f'SELECT * FROM "{eng.table_name}" LIMIT 200000').df()
    ds_id = resources.register(eng, df)
    S.datasources[ds_id] = {"name": name, "engine": "duckdb", "tables": eng.datasource().table_names}
    return {"id": ds_id, "tables": eng.datasource().table_names}


@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest, key_name: str = Depends(require_key),
            x_gemini_key: Optional[str] = Header(default=None)):
    if req.datasource_id not in S.datasources:
        raise HTTPException(404, f"unknown datasource_id {req.datasource_id}")
    _limit(key_name)
    conversation_id = req.conversation_id or uuid.uuid4().hex
    run_id = uuid.uuid4().hex[:16]
    S.runs[run_id] = {"conversation_id": conversation_id, "datasource_id": req.datasource_id,
                      "question": req.question, "status": "running", "key": key_name,
                      "started": time.time(), "result": None}
    S.counters["runs"] += 1

    config = {"configurable": {"thread_id": conversation_id}}
    existing = S.graph.get_state(config).values
    if existing and existing.get("datasource_id") != req.datasource_id:
        raise HTTPException(409, "a conversation is bound to one datasource")
    graph_input = orchestrator.turn_input(req.question, skip_storyteller=True)
    if not existing:
        graph_input |= {"schema": None, "datasource_id": req.datasource_id, "history": []}
    return StreamingResponse(_threaded(_run_stream(run_id, graph_input, x_gemini_key)),
                             media_type="text/event-stream")


@app.post("/v1/runs/{run_id}/resume")
def resume(run_id: str, req: ResumeRequest, key_name: str = Depends(require_key),
           x_gemini_key: Optional[str] = Header(default=None)):
    from langgraph.types import Command

    run = S.runs.get(run_id)
    if run is None or run["key"] != key_name:
        raise HTTPException(404, "unknown run")
    if run["status"] != "interrupted":
        raise HTTPException(409, f"run is {run['status']}, not waiting for input")
    _limit(key_name)
    run["status"] = "running"
    return StreamingResponse(_threaded(_run_stream(run_id, Command(resume=req.answer), x_gemini_key)),
                             media_type="text/event-stream")


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str, key_name: str = Depends(require_key)):
    run = S.runs.get(run_id)
    if run is None or run["key"] != key_name:
        raise HTTPException(404, "unknown run")
    return {"run_id": run_id, **{k: run[k] for k in ("conversation_id", "datasource_id", "question", "status", "result")}}


@app.post("/v1/runs/{run_id}/sql")
def edit_sql(run_id: str, edit: SQLEdit, key_name: str = Depends(require_key)):
    """Run user-edited SQL through the same guard. Edits are logged: an edit
    that the user keeps is a verified (question, SQL) pair for query memory."""
    run = S.runs.get(run_id)
    if run is None or run["key"] != key_name:
        raise HTTPException(404, "unknown run")
    engine, _ = resources.get(run["datasource_id"])
    df, err = engine.execute_query(edit.sql)
    EDITS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EDITS_LOG, "a") as f:
        f.write(json.dumps({"run_id": run_id, "question": run["question"],
                            "original_sql": (run.get("result") or {}).get("sql"), "edited_sql": edit.sql,
                            "ok": err is None, "ts": time.time()}) + "\n")
    if err:
        raise HTTPException(400, err)
    return {"result": _preview(df)}


@app.get("/v1/audit")
def get_audit(limit: int = 100, key_name: str = Depends(require_key)):
    """Audit trail of executed runs. Callers see only their own runs."""
    return audit.recent(limit=min(limit, 1000), api_key=key_name)


@app.get("/v1/metrics")
def metrics(_: str = Depends(require_key)):
    from core.semantic_cache import default_cache

    cache = default_cache()
    return {
        **S.counters,
        "semantic_cache": None if cache is None else {
            "hits": cache.stats.hits, "misses": cache.stats.misses,
            "hit_rate": cache.stats.hit_rate, "usd_saved": cache.stats.usd_saved,
        },
    }
