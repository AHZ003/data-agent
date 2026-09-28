# HTTP API

_Generated from the FastAPI app by `scripts/export_api_docs.py`; interactive docs at `/docs` on a running server._

Auth: `X-API-Key` header (keys from `DATAAGENT_API_KEYS`; off when unset). Optional `X-Gemini-Key` runs the request on the caller's own Gemini key.

| Method | Path | Summary |
|---|---|---|
| `GET` | `/healthz` | Healthz |
| `GET` | `/v1/datasources` | List Datasources |
| `POST` | `/v1/datasources` | Upload Datasource |
| `POST` | `/v1/analyze` | Analyze |
| `POST` | `/v1/runs/{run_id}/resume` | Resume |
| `GET` | `/v1/runs/{run_id}` | Get Run |
| `POST` | `/v1/runs/{run_id}/sql` | Run user-edited SQL through the same guard. Edits are logged: an edit that the user keeps is a verified (question, SQL) pair for query memory. |
| `POST` | `/v1/runs/{run_id}/feedback` | Thumbs up adds the answer to verified query memory; thumbs down queues it for review (python -m feedback.review), where it can become a golden case. |
| `GET` | `/v1/alerts` | Alerts |
| `GET` | `/v1/audit` | Audit trail of executed runs. Callers see only their own runs. |
| `GET` | `/v1/metrics` | Metrics |

## Server-Sent Events from `/v1/analyze` and `/resume`

| Event | Payload |
|---|---|
| `plan` | analysis types, steps, `clarifying_question` |
| `sql` | `{sql, cache_hit}` |
| `result_preview` | `{columns, rows (≤50), row_count}` |
| `chart_spec` | Plotly figure JSON |
| `validation` | critic status, confidence, warnings |
| `narrative_token` | `{text}`, streamed |
| `faithfulness` | `{warning}` when narrative numbers are not in the result |
| `clarify` / `confirm_cost` | the run paused; answer with `POST /v1/runs/{run_id}/resume` |
| `done` | `{run_id, conversation_id, error}` |
| `error` | `{run_id, message}` |
