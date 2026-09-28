# Results log

Every experiment gets a dated entry: what was run, the numbers, and what it
means. This is the source for the README scoreboard and the write-ups. Newest
first.

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

## Pending (needs a valid GOOGLE_API_KEY unless noted)

- Spider dev and BIRD mini-dev EX with CIs (`make bench-spider`, `make bench-bird`).
- Self-repair ablation with McNemar p (`spider-flash-single` vs `spider-flash-repair`).
- Retrieval ablations on BIRD (`bird-flash-*` rows in `benchmarks/ablations.yaml`).
- Chinook golden SQL baseline for the CI gate (`golden_sql_eval` + `gate --write-baseline`).
- Red team pipeline attacks, hardening off vs on (`python -m benchmarks.redteam.run`).
- Semantic cache hit rate / $ saved on a replayed workload.
- Model matrix + router + fine-tuned model (also needs Ollama and a GPU run).
- BigQuery golden set and bytes scanned with/without partition hints (needs GCP).

