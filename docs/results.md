# Results log

Every experiment gets a dated entry: what was run, the numbers, and what it
means. This is the source for the README scoreboard and the write-ups. Newest
first.

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

## Pending (needs a valid GOOGLE_API_KEY)

- Spider dev and BIRD mini-dev EX with CIs (`make bench-spider`, `make bench-bird`).
- Self-repair ablation with McNemar p (`spider-flash-single` vs `spider-flash-repair`).
- Retrieval ablations on BIRD (`bird-flash-*` rows in `benchmarks/ablations.yaml`).
- Chinook golden SQL baseline for the CI gate (`golden_sql_eval` + `gate --write-baseline`).
