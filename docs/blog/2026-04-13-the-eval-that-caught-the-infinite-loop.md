# The eval that caught the infinite loop

*How a 12-case golden set and 200 lines of tracing turned a silent
2,700-call bug into a 60-second diagnosis.*

---

## The bug that shouldn't have shipped

I was building DataAgent — a multi-agent LLM system that answers
questions about uploaded CSVs by running a LangGraph crew of six
specialized agents (planner, coder, critic, visualizer, predictor,
storyteller). The UI felt fine. Unit tests were green. I'd run maybe
thirty questions manually and nothing looked obviously broken.

Then I wired up an eval harness for the first time, pointed it at a
golden set of 12 questions across three datasets, and hit Run.

Case 1 finished in 11 seconds with a perfect score. ✅

Case 2 started at 22:04. At 22:19 — fifteen minutes later — it was
still going. The trace file had grown to 3.5 MB. I killed it.

When I pulled the traces apart I found something that made me stop
typing:

```
{
  "coder":       2703,
  "visualizer":  2703,
  "critic":      2703,
  "storyteller": 0,
  "run_id":      "a8f2..."
}
```

2,703 coder spans on a single question. Every single one errored with
HTTP 429. Gemini's free-tier quota had been exhausted within about
three minutes of case 2 starting, and the agent crew had been
*beating on the dead horse* for the twelve minutes after that.

This post is about what that bug actually was (three interlocking
failures, not one), what made it invisible for weeks, and why the
combination of an **eval harness** plus **structured tracing** turned
a four-hour debugging session into a sixty-second diagnosis.

## What you'd build in version one

When you're building an agent crew that calls an LLM, you learn fast
that the LLM will eventually fail. Usually a 429. Sometimes a
transient 500. Sometimes an actual content-policy block. The obvious
thing to do is wrap it in a retry loop:

```python
# agents/coder_agent.py — v1
def execute_analysis(question, schema, db):
    last_error = None
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            sql = _generate_sql(question, schema)
            return db.execute(sql)
        except Exception as e:
            last_error = str(e)
    return None, CodeResult(
        sql_query=last_error,   # ← bug 2
        success=False,
        error=f"Failed after {MAX_RETRY_ATTEMPTS}: {last_error}",
    )
```

And then, because the crew is coordinated by a LangGraph state machine,
you wire a retry edge from the critic back to the coder when
validation fails:

```python
# agents/orchestrator.py — v1
def should_retry(state):
    retry_count = state.get("retry_count", 0)
    if state["validation"]["status"] == "rejected" and retry_count < 3:
        return "retry"
    return "continue"
```

Both of these look fine. They pass code review. They pass unit tests
with mocked LLMs. They pass every manual smoke test you throw at them
because you only ever run the crew for thirty seconds before checking
the answer. And then they sit, latent, until you turn the crank for
twenty-five minutes.

## The three bugs

### Bug 1: the retry counter that was never written

`should_retry` reads `state["retry_count"]`. Nothing in the graph
*writes* it. It was initialized to 0 in `run_analysis` and stayed 0
forever. "Max 3 retries" was a dead letter — a guard checking a value
that can never change.

This is the kind of bug that's completely invisible under code review.
Reviewing `should_retry` in isolation, you see `retry_count < 3` and
nod. Reviewing `coder_node` in isolation, you don't think about it at
all because it's not one of its responsibilities. The bug lives in
the *edge* between the two, where no reviewer is looking.

I've started calling this the **dead-loop-guard pattern**: a
state-based bound that checks a variable nobody increments. In
review, pair any `state.get("foo") < N` with a search for the writer.
If you can't find one, the guard is a lie.

### Bug 2: errors masquerading as SQL

Look again at the retry loop in the coder:

```python
except Exception as e:
    last_error = str(e)
# ...
return None, CodeResult(
    sql_query=last_error,
    success=False,
    error=f"Failed after {MAX_RETRY_ATTEMPTS}: {last_error}",
)
```

When Gemini threw a 429, the coder caught it, stuck the exception
string into `last_error`, and then returned a `CodeResult` whose
`sql_query` field *was the 429 message*. Downstream consumers —
critic, visualizer — got a SQL query that read:

```
429 RESOURCE_EXHAUSTED. You exceeded your current quota.
```

The critic dutifully ran this through its "does the result answer the
question" check, saw an empty dataframe, and rejected it. Which fed
straight into Bug 1, which sent the request back to the coder, which
429'd again.

The lesson: **never let an exception string stand in for the thing it
failed to produce.** If a function can't compute X, it must not
return something shaped like X. Return `None`, raise, or tag the
result as invalid — but don't poison the type.

### Bug 3: no concept of terminal vs. transient

Even *if* the retry counter had been incrementing correctly, we'd
still have been retrying a quota-exhausted account three times per
question. That's not a bug, exactly, but it's the third ingredient in
the compound disaster: retrying a 429 is the *worst* thing you can
do. It doesn't recover the call. It counts against the per-minute
rate limit. It extends the outage. A 429 isn't a transient error for
the current session — it's a terminal one.

The orchestrator had zero notion of that distinction. Every rejected
validation was equally retry-able.

## The thing that made each bug individually invisible

Here's the part that took me longest to accept: **none of these bugs
would have been caught by unit tests as they're typically written.**

Unit tests for agent crews fall into two categories, both with the
same blind spot:

1. **Single-agent tests** that mock the LLM. These tell you whether
   the coder's prompt template renders correctly, whether the critic
   emits the right warnings on specific inputs, etc. They do *not*
   run the graph, so they can't see bugs in edges, conditional
   routing, or state management.
2. **Integration tests** that run the graph but with a single
   hand-picked question, inspected manually. You ran it, you saw the
   answer, you moved on. You never sat there for twenty-five minutes
   waiting to see what happens when Gemini revokes your access.

The thing that catches crew-level loop bugs is running the crew
**unattended, against a diverse input set, for long enough that the
pathological path gets a chance to fire.** That's exactly what an
eval harness is.

## The harness that caught it

The eval harness is 200 lines of Python plus a YAML file of golden
cases. Each case is a (dataset, question, expected) triple. Each
case runs the full crew end-to-end, no mocks. The scorer checks
eight dimensions — SQL keywords, top-row value, row-count bounds,
chart type, narrative faithfulness (LLM-judged), critic confidence —
and computes a weighted score. The runner aggregates per-case,
per-dimension, and per-tag, then writes a `results.json` and a
`report.md`.

Here's the thing I didn't expect: **the very first time I ran it
end-to-end, it caught a bug.** Case 1 passed. Case 2 hung. Case 3
through 12 never ran.

If I hadn't had the harness, the bug would have shipped. It would
have shipped exactly the way you'd expect a latent bug to ship — as a
user complaint several weeks later saying "I asked it a question and
it just spun forever and then billed me $40." Instead it showed up
as a failed run in my own terminal, on my own branch, before any
deploy.

## The tracing that explained it

Having the harness fail loudly was necessary but not sufficient.
After I killed the run, I still had to figure out *why*. That's
where the tracing module — the other new thing I'd built that cycle
— paid off.

Every agent node in the orchestrator emits a structured JSON span
with `run_id`, `agent_name`, `duration_ms`, `tokens_in`, `tokens_out`,
`cost_usd`, `status`, and arbitrary metadata. On a working run for a
typical question, that's 6 spans: one per agent. On the broken run I
got a file with **8,109 lines** — 2,703 per retry-looping agent.

From there, the diagnosis was a one-liner:

```python
>>> from collections import Counter
>>> Counter(span["agent_name"] for span in tracing.read_spans(run_id=rid))
Counter({'coder': 2703, 'visualizer': 2703, 'critic': 2703, ...})
```

I didn't have to read 8,000 lines of prose logs to find the pattern.
I didn't have to instrument the agent crew from scratch to capture
what it did. I had a *queryable trace store*, and the trace store
told me the shape of the bug in one `Counter` call.

If I'd only had one of these — eval harness without tracing, or
tracing without an eval harness — the bug would have taken hours to
unwind. With both, it was sixty seconds.

## The fixes

I'm going to be brief here because the interesting part of this post
isn't the patch:

1. **Increment `retry_count` inside `coder_node`**, on every entry to
   the node. The bound now actually bounds. One line.
2. **Detect quota errors in `coder_agent`** via a small helper
   `_is_quota_error(exc)` that scans exception messages for the
   known quota phrasings (`RESOURCE_EXHAUSTED`, `429`, `rate limit`,
   etc). On a match, short-circuit out of the retry loop with a
   sentinel `QuotaExhausted: <msg>` string — don't poison `sql_query`.
3. **Short-circuit `should_retry` on the sentinel.** A quota-exhausted
   failure is terminal; the graph should move to the storyteller,
   which will surface a user-friendly "we hit our AI quota" message,
   not loop back to the coder.
4. **Seven regression tests** that would have prevented the incident:
   the most important one patches `_generate_sql` to raise a 429 and
   asserts the coder makes *exactly one* call, not
   `MAX_RETRY_ATTEMPTS`. There's also a full-loop test with a
   `while iterations < 10` safety net that would have hit the net
   and failed under the pre-fix code.

Total code change: about 40 lines. The post-mortem is longer.

## Lessons I'm taking forward

1. **Eval harnesses are load-bearing infrastructure.** I used to
   think of them as a "should have, eventually" item — the kind of
   thing you add after you ship. No. Build the harness first, or at
   least second. The dormant-infinite-loop bug lived in a codebase
   with 18 passing unit tests. It died the first time an eval run
   forced the crew to answer 12 questions in sequence.
2. **Tracing is what turns "mystery" into "diagnosis."** The
   operational lesson is not "always add more logs." It's that
   *structured*, *queryable*, *correlated* traces let you ask
   questions of a failing run the way you'd ask questions of a
   database. If you've ever debugged a multi-agent LLM system
   without this and thought "these things are impossible to debug,"
   I promise you they aren't. They're just impossible to debug
   *without the right tooling*.
3. **Dead-loop-guards are a code-review antipattern.** When you see
   `state.get("foo") < N`, search for the writer. If you can't find
   one, the guard doesn't exist — you just think it does.
4. **Never `except Exception:` inside a retry loop without a
   "terminal?" check.** A catch-all handler with no sense of which
   errors are retryable is how you turn a 3-second outage into a
   25-minute quota holocaust.
5. **Observability and evaluation are multiplicative.** They don't
   add value linearly. Each one alone is okay. Together they
   convert debugging time from hours to seconds, and they're both
   cheap to build. The eval harness is 200 lines. The tracing
   module is another 200. For 400 lines of infrastructure code, you
   get an order-of-magnitude improvement in how fast you can find
   and fix the next bug.

## One last thing

The nastiest bit of this whole incident, the thing that stuck with
me most after writing it up, is this: **the eval harness caught the
bug on its first-ever run**. Every day I didn't have the harness,
the bug was there. Every time I manually clicked through a demo and
congratulated myself on shipping a smooth flow, the bug was there.
Every passing unit test run, the bug was there.

The first time I actually let the crew run unattended — one evening,
on a laptop, against a set of questions I'd typed up in a YAML file
— the bug surfaced inside 180 seconds.

The moral I'm taking away is not "write more tests." It's "let the
system run unsupervised against diverse inputs as early as you can,
because the failure modes that matter are the ones that don't fire
during hand-testing."

Build the harness first.
