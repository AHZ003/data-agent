# Results log

Every experiment gets a dated entry: what was run, the numbers, and what it
means. This is the source for the README scoreboard and the write-ups. Newest
first.

## 2026-10-22 — BigQuery NYC taxi golden set (Gemini 3.8 Flash)

- 40 questions over a partitioned + clustered copy of 2022 yellow trips (36.3M
  rows): EX 75.0% [60.0, 87.5] strict, 97.5% [92.5, 100] allowing extra columns
  (9 of 10 strict misses add the ranked-by measure on purpose). One real miss: a
  zone-name literal the value index would have caught.
- Partition hints removed: 72.5% / 95.0%, McNemar p = 1.0; identical bytes on
  date-range questions. The model filters on the date the question names either way.
- Layout: the same SQL scans 90% less on date-range questions than on the
  unpartitioned public table (0.060 vs 0.607 GiB), 3% less on whole-year ones.
- Cost guard did not fire at the default 10 GiB; at the demo's 1 GiB, 2/40
  ask for confirmation. Total spend: < $0.06 BigQuery, ~$0.50 Gemini.
- Found and fixed: the BigQuery extra lacked `db-dtypes` (every live query would
  have failed) and Docker didn't install the extra; BigQuery API calls had no
  request timeout (an eval run hung for 30+ minutes); the public zone table
  repeats two zones (joins double-counted).

## 2026-10-21 — Spider, BIRD, retrieval, models, router, red team (Gemini 3.8 Flash)

All on 200-question difficulty-stratified subsets (seed 0), repair mode unless
noted, 95% bootstrap CIs, paired exact McNemar. Total spend for the day ≈ $23.

- **Spider dev:** 86.0% [81.0, 90.5], $0.0049/q. Self-repair vs single:
  +0.5 [−1.0, +2.5], p = 1.0 (1/2 discordant).
- **BIRD mini-dev:** evidence on 72.5% [66.0, 78.5]; repair +0.0 (5/5
  discordant); evidence off 55.0% [48.0, 62.0], −17.5 [−23.5, −11.5],
  p = 7.9×10⁻⁸ (40/5). $0.011–0.014/q.
- **Retrieval (BIRD):** values +1.0 (p .73), schema k=30 −1.0 (p .79), both
  −0.5 (p 1.0), memory 0.0 (p 1.0), all combined 76.0%, +3.5 [−0.5, +7.5],
  p = 0.14 (5/12). See `docs/retrieval.md`.
- **Models (Spider):** Flash-Lite 82.0% at $0.00053 (p95 1.4 s); 3.8 Flash 86.0%
  at $0.0049 (p95 43.6 s); 3.1 Pro 88.0% at $0.0111. Neither Δ significant.
- **Router (held-out n=100):** Lite→Pro dominated by plain Flash; Lite→Flash
  reaches Flash's 84% at 59% of its cost. `docs/models.md`, `docs/pareto_spider.svg`.
- **Red team, pipeline:** 1/18 attacks succeed with hardening off, 0/18 on.
  Two scoring/robustness fixes first: canaries are judged on the narrative only
  (tables echo poisoned data by design), and a storyteller crash on runs with
  no result.
- **Ops findings:** a stalled Gemini request hung a run (no SDK timeout → 120 s
  timeout added); a 503 from the embedding API aborted a run (transient errors
  now retried, per-question failures isolated); laptop sleep inflated four
  latencies (rows re-measured; runs now use `caffeinate`).

## 2026-10-20 — Chinook golden SQL set, Gemini 3.8 Flash (first live run)

`python -m benchmarks.golden_sql_eval --concurrency 4` (repair mode, n=105):
**EX 97.1% (95% CI 93.3–100.0%)**, $0.00435/question ($0.46 total), p50 4.4 s,
p95 8.9 s. Saved as the CI gate baseline (`benchmarks/baselines/golden_sql.json`).

- The first run scored 95.2%: 3 SQL answers were cut off mid-statement because
  Gemini 3.x thinking tokens count against `max_output_tokens` (983 of 1,024 spent
  thinking on one question). Output budget raised to 8,192 (billing is per token
  used) and truncation is now flagged per example; rerun had 0 truncations. With
  n=105 the 95.2 → 97.1 change is not itself a significant result.
- Remaining misses (3): two return the correct answer plus an extra column
  (official Spider rule counts these wrong: output_shape), one uses integer
  division (`/ 60000` instead of `/ 60000.0`) despite the dialect note.
- Gemini 2.5 models return 404 for new API projects; default is now 3.8 Flash.

## 2026-09-28 — Eval power analysis

`python -m benchmarks.power`: paired MDE at 80% power is 6.3–8.9 pts on n=200
(10–20% discordance) and 2.8–3.9 pts on Spider dev's 1,034. Two identical
configs (8% answer flips) look ≥2 pts apart 51% of the time on n=200; McNemar
false-positive rate 3.7%. Takeaway: nightly 200-question subsets track
regressions of ~7+ pts; smaller claims need the full dev set and the paired test.

## 2026-09-28 — Load test, stubbed LLM (service overhead)

`loadtest/`: 0 requests completed at 10 users → deadlock on a shared SQLite
connection (GIL / connection-mutex inversion in sqlite3 callbacks); fixed by
serializing engine connections. Then throughput capped at ~23 req/s → SSE
generators pinned Starlette pool threads; fixed with an asyncio-queue bridge.
200 users: 23.2 → 32.3 req/s, p50 6.1 → 3.9 s, 0 failures. Details:
`docs/performance.md`.

## 2026-09-28 — SQLite vs DuckDB

`python -m benchmarks.engine_bench --rows 1000000 10000000`: DuckDB 15–74× faster
at 1M rows, 30–165× at 10M (group-by sum 2,169 → 16 ms). Load 6.2 s vs 2.8 s.

## 2026-09-28 — Red team, SQL layer (offline)

`python -m benchmarks.redteam.run --offline`: 23 guard-evasion attacks; original
regex guard on a bare connection let 5 through (2 unbounded queries, 3 DuckDB
file reads), current stack 0. Legitimate queries wrongly blocked: 3/8 → 0/8.

## 2026-09-28 — Schema-linking recall (offline)

`python -m benchmarks.retrieval_eval --benchmark {spider,bird} --embedder {none,hashing} [--values]`

- Spider dev, BM25, k=20: all gold columns kept for 97.5% (96.5–98.4) of 992 questions; prompt 84% of full.
- BIRD mini-dev, BM25, k=30: 89.6% (86.8–92.2) of 500; prompt 61%. With value hints: 90.2%.
- Hashing embedder: no gain over BM25 on either benchmark.
- Takeaway: pruning is only worth testing end-to-end on BIRD; the risk is ~10% of
  questions losing a needed column at k=30. Details in `docs/retrieval.md`.

## 2026-09-28 — Harness oracle checks

Gold SQL fed through the production execution path and matchers:

- Spider dev 1034/1034, Chinook golden set 105/105.
- BIRD mini-dev 498/500; both misses are gold queries slower than BIRD's
  official 30s timeout on a laptop (count as misses for every config).
- Found and fixed: pandas NULL→NaN made any result containing NULL unmatchable.

## Pending

- Fine-tuned 1.5B model and local Qwen-7B on the frontier (GPU run, Ollama).
- Full 500-question BIRD run for the combined-retrieval config (more power).
- Semantic cache hit rate on a replayed workload; real-model load test.
