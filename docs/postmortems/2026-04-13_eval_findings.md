# Postmortem — 2026-04-13: Eval harness surfaces infinite retry loop

**Status:** resolved
**Severity:** would-be-SEV-1 in production
**Author:** DataAgent team
**Detected by:** first full run of `benchmarks/runner.py`
**Runtime of the incident:** ~25 minutes before manual abort
**Customer-visible impact:** none (caught pre-deploy)
**Cost impact:** burned ~2,700 Gemini calls on 2 test cases — would have
been the entire daily quota in a production deploy

## Summary

The evaluation harness (`make eval-fast`) was run for the first time
end-to-end against the live agent crew. Two cases into the run it had
already emitted **2,703 spans from the `coder` agent** — vs. an expected
~6 — and the run was making no forward progress. Diagnosis revealed
three independent bugs which, together, turn any persistently-rejecting
critic path into a wedged infinite loop that quietly burns LLM quota.

All three bugs existed in the codebase before the eval harness was
built; they were latent and invisible until the harness forced the
crew to run unattended across a diverse question set.

## Timeline (UTC)

| Time  | Event |
|-------|-------|
| 22:01 | Full eval started (`python -m benchmarks.runner --no-judge`) |
| 22:01 | First case `ss_top_region_by_sales` — completes in 11.2s, score 1.00 ✅ |
| 22:04 | Case 2 begins; makes no forward progress after 30s |
| 22:13 | Trace file has grown to 3.5 MB with 8k+ lines |
| 22:19 | First full eval aborted manually after >15 min stall |
| 22:19 | Re-run started with direct file logging (no grep pipeline) |
| 22:45 | Second run aborted after case 2 hit 1,391 s (23 min!) on one case |
| 22:46 | Trace inspection reveals 2,703 coder spans, 100% errored with 429 |
| 22:50 | Root cause identified |
| 23:10 | Fix + regression tests landed; `pytest` green (49/49) |

## Root cause

Three bugs compound into the observed behavior:

### Bug 1 — `retry_count` is read but never written

`agents/orchestrator.py:should_retry` gates retries on `retry_count < 3`,
but nothing in the graph ever incremented the counter. The field was
initialized to `0` in `run_analysis` and stayed `0` forever. The
"maximum 3 retries" guardrail was a dead letter.

```python
# BEFORE
def should_retry(state):
    retry_count = state.get("retry_count", 0)   # always 0
    if status == "rejected" and retry_count < 3: # always True
        return "retry"
    return "continue"
```

Whenever the critic returned `rejected`, the graph looped back to the
coder *unboundedly*.

### Bug 2 — Coder agent swallows LLM errors as "SQL"

`agents/coder_agent.execute_analysis` wraps `_generate_sql(...)` in a
bare `except Exception`, stashes `str(e)` into a `last_error` variable,
and retries. When Gemini returns a 429, the coder:

1. Catches the 429 exception.
2. Stores the error string in `last_error`.
3. Retries — which 429s again.
4. After `MAX_RETRY_ATTEMPTS` (3), returns a `CodeResult` where
   **`sql_query` itself contains the error string**:

```python
# BEFORE
return None, CodeResult(
    sql_query=last_error,  # ← error message, not SQL
    success=False,
    error=f"Failed after {MAX_RETRY_ATTEMPTS} attempts. Last error: {last_error}",
)
```

The downstream Critic saw an empty `result_df` and marked the case
rejected — feeding directly into Bug 1 and restarting the coder. On the
next loop, the coder's Gemini quota was *still* exhausted, so it 429'd
again, and the orchestrator retried again, indefinitely.

### Bug 3 — No terminal-error short-circuit in the retry edge

Even *if* the retry counter had been incrementing correctly, a 429 is
not a meaningful retry condition — backing off and retrying a
quota-exhausted account burns more quota and extends the outage. The
orchestrator had no concept of terminal vs. transient errors. Every
rejection was treated as equally retry-able.

## Blast radius

- **Trace emission:** 2,703 coder spans + 2,703 visualizer spans +
  2,703 critic spans in ~25 min = ~8,100 JSON trace lines, 3.5 MB on
  disk. A longer run would have filled the daily trace file indefinitely.
- **LLM spend:** Gemini free-tier quota exhausted within the first few
  minutes. The agent crew then kept calling the API anyway, each call
  returning 429 immediately but still counting toward per-minute rate
  limits. A paid deployment would have billed ~2,700 input-token calls
  plus some fraction of successful calls from earlier in the run.
- **User-facing time:** each affected analysis would appear to hang
  from the user's perspective, with no progress indication beyond
  "agents working…" in `st.status`.

## Detection

The eval harness (`benchmarks/runner.py`) found this. It's the first
thing we ever built that runs the agent crew unattended across
multiple diverse questions, so it was the first thing that could
observe the pathological loop. None of the existing unit tests would
have caught this because they mock the LLM calls or exercise single
agents in isolation.

The *structured tracing module*
(`core/tracing.py`, also new this cycle) made diagnosis fast: rather
than reading 8k log lines, we queried the JSONL trace file and
immediately saw the `{coder: 2703, visualizer: 2703, ...}` span
histogram that made the bug obvious.

## Fixes

All three bugs are fixed in a single commit:

### Fix 1 — increment `retry_count` in `coder_node`

```python
# agents/orchestrator.py
def coder_node(state):
    state["retry_count"] = state.get("retry_count", 0) + 1
    ...
```

This is the minimal-invasive fix — the counter now increments every
time the graph enters the coder node, which includes both the first
call and every retry-edge re-entry.

### Fix 2 — detect quota errors in `coder_agent`

A new module-level helper `_is_quota_error()` scans exception
messages for known quota phrasings (`RESOURCE_EXHAUSTED`, `429`,
`rate limit`, etc.). When a quota error is caught, the coder
immediately returns a `CodeResult` with an error string prefixed by
a new `QUOTA_SENTINEL = "QuotaExhausted"` constant. The agent no
longer retries on these errors, and no longer pollutes `sql_query`
with the error string.

### Fix 3 — `should_retry` short-circuits on the sentinel

```python
# agents/orchestrator.py
def should_retry(state):
    if "QuotaExhausted" in (state.get("error") or ""):
        return "continue"
    ...
```

A quota-exhausted error is a terminal failure. The graph now skips
the retry edge and proceeds to `storyteller`, which will surface a
user-friendly "we hit our AI quota, try again later" message instead
of hanging.

## Regression tests (`tests/test_retry_and_quota.py`)

Seven tests pin the fixes in place. The two most important:

- **`test_coder_shortcircuits_on_429`** — patches `_generate_sql` to
  raise a 429, asserts the coder makes *exactly one* call (not
  `MAX_RETRY_ATTEMPTS`), and asserts the resulting `CodeResult.error`
  contains the `QuotaExhausted` sentinel. This is the single test
  that would have prevented the incident.
- **`test_retry_bound_regression_full_loop`** — simulates the exact
  pathological case: the coder always succeeds at the SQL level but
  the critic always rejects. The test runs the coder/should_retry
  loop in a `while` with an `assert iterations < 10` safety net, and
  asserts that `retry_count` ends at exactly 3. Before Fix 1, this
  test would have hit the safety net and failed; after the fix, it
  terminates after 3 iterations.

Plus 5 more covering variant quota message phrasings, the
should_retry decision table, and the guarantee that `sql_query` is
no longer overwritten by error strings.

Total test suite: **49 passing** (18 original + 24 SQL-guard/tracing + 7 retry/quota).

## What we learned

1. **Eval harnesses are not a nice-to-have.** The entire incident was
   dormant for weeks until a golden-set runner forced the crew to run
   unsupervised. Unit tests on individual agents passed. Manual
   ad-hoc testing passed. Only the harness surfaced the bug — and it
   did so on its *first full run*. Build the harness first.
2. **Observability is what turns "mystery" into "diagnosis."** Without
   `core/tracing.py` emitting structured JSONL spans, reconstructing
   "what did the crew actually do during those 25 minutes" would
   have required reading 8k lines of prose logs or instrumenting
   the crew from scratch. The trace file made the root cause
   obvious in one `jq`/Python one-liner.
3. **Never `except Exception:` and then continue a retry loop.**
   Every catch-all exception handler inside a retry loop needs at
   least one "is this terminal?" check. Applied here as
   `_is_quota_error()`, but the rule is general.
4. **Guards that check state must be paired with writers of that
   state.** `retry_count < 3` is a classic dead-loop-guard — a value
   that's *only* read and never written will silently fail open.
   In code review, pair any `state.get("foo") < N` check with a
   search for where `foo` is written. If it isn't, the guard is a lie.
5. **Tracing + eval are compounding investments.** The eval caught
   the bug. Tracing explained it in seconds. Together they turn
   "multi-agent LLM systems are hard to debug" into "multi-agent
   LLM systems are hard to debug *without this tooling*." Every
   hour spent on either of these pays back 10x in incident time.

## Action items

- [x] Fix `retry_count` never-incremented bug (orchestrator.py)
- [x] Fix coder-agent silent 429 swallowing (coder_agent.py)
- [x] Add quota-aware short-circuit in `should_retry`
- [x] Add regression tests (`test_retry_and_quota.py`)
- [x] Full pytest suite green (49/49)
- [x] Write this postmortem
- [ ] Re-run the full eval (`make eval-fast`) once Gemini quota resets
      and commit the resulting `outputs/eval/report.md`
- [ ] Add a pytest marker for "unsupervised-loop" tests that run the
      agent crew with a mocked LLM over multiple turns, so future
      loop bugs surface in CI and not in eval
- [ ] Emit a warning in the UI when `QuotaExhausted` is hit so users
      know to retry later instead of hammering the button
