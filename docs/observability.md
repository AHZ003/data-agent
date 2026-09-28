# Observability and the feedback loop

## Traces

Every graph node and LLM call is a span in `outputs/traces/<day>.jsonl`
(`core/tracing.py`): run id, agent, model, duration, tokens in/out, cost, and
node metadata (SQL, row count, cache hit, …). Token counts come from the
provider's usage metadata; thinking tokens count as output.

With `OTEL_EXPORTER_OTLP_ENDPOINT` set (and `uv sync --extra otel`), the same
spans are exported over OTLP (`core/otel.py`) nested under a `dataagent.run`
root span, with GenAI semantic-convention attributes. For Langfuse:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic $(printf '%s' "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" | base64)"
```

The JSONL file remains the source of truth; the OTLP export is a mirror, so a
collector outage never fails a request.

## Online evaluation

`core/online_eval.py` scores a sample of live API runs
(`DATAAGENT_ONLINE_EVAL_RATE`, default 10%): executed, non-empty, critic not
rejecting, narrative numbers supported by the result, and optionally the LLM
faithfulness judge (`DATAAGENT_ONLINE_JUDGE=1`). Records land in
`outputs/online_eval.jsonl`.

## Alerts

`monitoring/alerts.py` (CLI, or `GET /v1/alerts`) compares the last 24 hours
with the previous 7 days:

| Signal | Fires when |
|---|---|
| Pass rate | the current window's upper 95% bound is below the baseline rate (same CI-aware rule as the CI gate) |
| p95 latency | above 1.5× the baseline p95 (`ALERT_LATENCY_RATIO`) |
| Cost per question | above 1.5× the baseline mean (`ALERT_COST_RATIO`) |

With fewer than 20 scored runs in either window it reports "insufficient data"
rather than guessing. Schedule it with Cloud Scheduler calling `/v1/alerts` and
notify on a non-empty `alerts` list.

## Feedback loop

```
thumbs up   ─► user query memory ─► few-shot examples for similar questions
thumbs down ─► review queue ─► reviewer fixes SQL ─► benchmarks/golden_user.yaml ─► golden eval + CI gate
```

- `POST /v1/runs/{id}/feedback {"rating": "up"|"down", "comment": ...}`
- `python -m feedback.review list | promote <run_id> --sql '...' | stats`
- Promotion validates that the gold SQL runs and returns rows on the sample
  database, and records the originating `run_id`.
- `GET /v1/metrics` → `feedback.golden_from_feedback` counts how many golden
  cases came from real usage.
- User memory is enabled in production with
  `DATAAGENT_RETRIEVAL="memory_k=3,embedder=auto"` (plus any other retrieval
  flags); it carries the same leakage guard as benchmark memory.

Pending: the Streamlit app's thumbs-up/down buttons (the API has them); see the
`app.py` notes in the project status.
