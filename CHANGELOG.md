# Changelog

## v3.0 — models, service, safety, distribution (2026-09-28)

- FastAPI service: SSE streaming, multi-turn conversations (LangGraph
  checkpointer), clarify / cost-confirm interrupts, editable SQL, auth + rate
  limits, semantic cache, feedback, audit, alerts. Load-tested; two
  concurrency bottlenecks found and fixed.
- Engines: DuckDB and BigQuery (dry-run cost guard) next to SQLite.
- Safety: PII masking and local-only mode, prompt-injection hardening, numeric
  faithfulness check, red-team suite, system card.
- Models: LLM gateway (Gemini / Ollama per agent), difficulty router,
  fine-tuning data + QLoRA notebook, cost/accuracy frontier.
- Observability: OTLP export (Langfuse), online eval, drift alerts, feedback →
  memory / golden cases.
- Distribution: MCP server, CLI, docs site, generated API reference.

## v2.0 — evaluation platform and retrieval (2026-09-28)

- Shared text-to-SQL harness on the production coder path; Spider + BIRD with
  official matching; oracle checks; bootstrap CIs, McNemar, CI-aware gate;
  ablation runner; eval dashboard; 105-question Chinook golden set; error
  taxonomy tooling.
- Retrieval: hybrid schema linking, value retrieval, query memory (train
  only), semantic layer; offline recall evaluation.
- Cloud Run deployment with Workload Identity Federation.

## v1.1 — foundations (2026-09-28)

- Multi-table `DataSource`; three-layer SQL guard (sqlglot AST, sqlite3
  authorizer, timeout); schema agent as a graph node; generated graph diagram;
  uv lockfile; dead code removed; real token/cost accounting.

## v1.0 — portfolio demo (2026-04)

- Seven-agent LangGraph pipeline, Streamlit UI, JSONL tracing, 18-case eval
  harness, prompt registry, postmortem.
