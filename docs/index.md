# DataAgent

A text-to-SQL analytics agent built to be measured: ask a question about a
workspace (CSV/Parquet, SQLite, DuckDB, BigQuery), get the SQL that ran, the
result, a chart, and a short narrative — through a pipeline whose every change
is evaluated with confidence intervals and paired significance tests.

- **Same code path everywhere.** The Streamlit app, the HTTP API, the CLI, the
  MCP server and every benchmark run `coder_agent.generate_and_execute`.
- **Guarded execution.** sqlglot AST allow-list, engine-level read-only
  enforcement, timeouts, BigQuery dry-run cost guard. Red team: 0/23 SQL
  guard-evasion attacks succeed (5/23 against the original regex guard).
- **Evaluation you can trust.** Official Spider/BIRD matching, harness oracle
  checks (Spider 1034/1034), bootstrap CIs, exact McNemar tests, a CI gate that
  ignores noise.

Start with the [Quickstart](quickstart.md), or read the
[evaluation methodology](evaluation.md).
