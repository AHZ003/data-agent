# DataAgent system card

_Last updated 2026-09-28. Numbers marked **pending** need a run with a valid
Gemini API key; see `docs/results.md` for the dated log._

## What it is

A text-to-SQL analytics agent. A user asks a question in plain English about a
workspace (uploaded CSV/Parquet files, a SQLite database, or BigQuery tables);
a LangGraph pipeline plans, writes and runs one read-only SQL query, checks the
result statistically, draws a chart, and writes a short narrative.
Default model: Gemini 2.5 Flash.

## Intended use

- Exploratory analytics by people who can check the SQL it shows them.
- Internal dashboards and ad-hoc questions over non-critical data.
- A reference implementation of an evaluated, guarded LLM data tool.

## Not intended for

- Decisions that need audited numbers without a human reviewing the SQL.
- Writing to data: every engine is read-only by construction.
- Regulated data (health, payment) without local-only mode and a legal review.
- Free-text documents: PII detection is tuned for structured columns.

## How data leaves the machine

The LLM provider receives the question, the schema (names, types, keys), up to
three sample values per column, and up to 20 result rows for the narrative.

| Control | Effect |
|---|---|
| PII masking (`core/pii.py`) | Columns detected as PII (by name or validated value patterns: email, phone, SSN, Luhn-valid cards, IP, IBAN) show `<EMAIL>`-style placeholders instead of samples, and are masked in narrative inputs |
| `DATAAGENT_LOCAL_ONLY_VALUES=1` | No raw value reaches the LLM: no samples, no value hints; the narrative sees only row counts and numeric summaries |
| Prompt-injection hardening (`core/injection.py`) | Data is wrapped in `<untrusted_data>` with an explicit notice; instruction-like values are replaced before any prompt is built |
| Audit log (`core/audit.py`, `GET /v1/audit`) | Every API run: who, question, executed SQL, engine, bytes scanned, PII columns touched |

## Safety of execution

| Layer | SQLite | DuckDB | BigQuery |
|---|---|---|---|
| AST guard (sqlglot): one SELECT, no DML/DDL/PRAGMA/ATTACH, no file-reading functions or path-like tables | yes | yes | yes |
| Engine-level read-only | sqlite3 authorizer | external access off + config locked; read-only files | dry run first; `maximum_bytes_billed` |
| Timeout | progress handler | interrupt timer | job timeout |
| Cost control | — | — | refuse above `BQ_MAX_BYTES`; ask above `BQ_CONFIRM_BYTES` |

## Evaluation

| What | Result |
|---|---|
| Harness correctness (gold SQL through the production path) | Spider 1034/1034, Chinook 105/105, BIRD 498/500 (2 gold queries exceed BIRD's 30 s limit) |
| Red team, SQL guard evasion (23 attacks) | before 5/23 succeeded (22%), after **0/23**; legitimate queries wrongly blocked: before 3/8, after 0/8 |
| Red team, data-borne prompt injection (18 attacks, 4 categories) | **pending** (`python -m benchmarks.redteam.run`) |
| Schema-linking recall, all gold columns kept | Spider 97.5% at k=20; BIRD 89.6% at k=30 (offline, BM25) |
| Spider dev / BIRD mini-dev execution accuracy | **pending** |
| Chinook golden SQL set (105 questions) | **pending** |
| Service overhead (stubbed LLM, 1 process) | 0 failures to 200 users; p50 at the model-time floor up to ~100 users |

## Known failure modes and limits

- **Wrong but plausible SQL.** The agent can answer a slightly different
  question than asked (wrong join path, wrong aggregation grain). The SQL is
  always shown; the critic flags some statistical issues but cannot know intent.
- **Ambiguity.** The planner asks a clarifying question only when no sensible
  default exists; otherwise it silently picks one (e.g. "top customers" → by
  total spend).
- **Schema pruning can drop a needed column** on wide schemas (~10% of BIRD
  questions at k=30). Pruning is off unless configured.
- **PII detection misses** names and places inside free-text cells and
  unusually named PII columns. Presidio/NER was not used, to keep the image
  small; use local-only mode for sensitive data.
- **Prompt injection** is reduced, not eliminated: neutralizing is pattern
  based, and a novel phrasing can pass. Output checks (numeric faithfulness)
  catch planted numbers but not planted prose.
- **Numeric faithfulness** checks numbers against the result table; derived
  numbers it cannot reproduce (e.g. a ratio of two cells) are flagged as
  unsupported even when correct.
- **Conversations** persist only with the Postgres checkpointer; the default
  SQLite checkpointer is per instance.

## Change control

Prompts are versioned files (`prompts/`); every eval run stamps prompt SHAs.
CI gates PRs on the Chinook golden set with a CI-aware regression gate. The
red-team guard cases run as unit tests on every commit.
