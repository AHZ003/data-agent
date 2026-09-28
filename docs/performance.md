# Performance

## Load test (service overhead, simulated LLM)

`loadtest/serve_stub.py` serves the real API (graph, guarded SQL on the Chinook
sample, checkpointer, SSE) with every LLM call replaced by a fixed 800 ms sleep.
It measures what the service costs under concurrency without spending quota.
The floor is 2.4 s per request: three simulated model stages (plan, SQL,
streamed narrative). `loadtest/locustfile.py` asks a realistic question mix,
reads the full SSE stream, and asks follow-ups in the same conversation 20% of
the time. Single uvicorn process on an Apple Silicon laptop, SQLite
checkpointer, 45 s per level.

### Bottleneck 1: deadlock (fixed)

The first run completed **0 requests at 10 users**: every request hung. Thread
dumps (`faulthandler`, `kill -USR1`) showed graph threads inside
`Database.execute_query` on the sample workspace's single SQLite connection,
which all conversations share. sqlite3's progress-handler and authorizer
callbacks need the GIL while holding the connection mutex, and another thread
holding the GIL was blocked on `set_progress_handler` (which needs that mutex).
Classic lock-order inversion. Fix: each engine serializes use of its
connection with a lock, and the row cap became a per-call argument instead of
mutable shared state. Regression test:
`tests/test_sql_guard.py::test_concurrent_queries_on_one_connection_do_not_deadlock`.

### Bottleneck 2: SSE streams pinned server threads (fixed)

After the deadlock fix, throughput stopped growing at ~23 req/s and latency
climbed with users. Each SSE response was a sync generator, which Starlette
pulls through its 40-thread pool; every open stream held a pool thread for its
whole 2–3 s run. Fix: the graph runs on its own thread and pushes events onto
an `asyncio.Queue` that the response awaits, so waiting streams hold no
threads (`api/main.py::_threaded`).

| Users | Before: p50 / p95 | Before: req/s | After: p50 / p95 | After: req/s | Failures |
|---|---|---|---|---|---|
| 50 | 2.4 / 2.6 s | 11.5 | 2.4 / 2.8 s | 11.9 | 0 |
| 100 | 2.8 / 3.3 s | 20.8 | 2.4 / 3.1 s | 22.7 | 0 |
| 200 | 6.1 / 8.3 s | 23.2 | 3.9 / 4.4 s | 32.3 | 0 |

Up to ~100 concurrent users the service adds almost nothing over model time.
Past that a single process saturates again; the next suspect is the SQLite
checkpointer, which funnels every checkpoint write through one connection and
lock. Production uses the Postgres checkpointer with a connection pool
(`DATAAGENT_CHECKPOINT_DSN`) and Cloud Run scales out processes, so this limit
is per instance.

Reproduce:

```bash
RATE_LIMIT_PER_MINUTE=1000000 RATE_LIMIT_BURST=1000000 uv run python loadtest/serve_stub.py --port 8010 &
uvx locust -f loadtest/locustfile.py --host http://127.0.0.1:8010 --headless -u 100 -r 20 -t 45s
```

The rate limit is raised because every locust user shares one (anonymous) key.

## Semantic cache

`core/semantic_cache.py` (enable with `DATAAGENT_SEMANTIC_CACHE=1`) reuses SQL
for questions whose embedding is within cosine 0.95 of a cached one on the same
schema version, re-executing it on current data and skipping the coder's LLM
calls. `GET /v1/metrics` reports hits, misses, hit rate and dollars saved.

Pending (needs a valid API key): hit rate and $ saved on a replayed workload of
real question paraphrases.

## Real-model latency

Pending (needs a valid API key): the same locust run against a real deployment.
The simulated run bounds service overhead; real latency is dominated by Gemini.
