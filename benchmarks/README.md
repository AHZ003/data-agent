# DataAgent — Evaluation

Three layers of evaluation, all running the production code path:

| Suite | What it measures | Size | Where it runs |
|---|---|---|---|
| **Chinook golden SQL** (`golden_sql_eval.py`) | Execution accuracy on a real 11-table schema, tagged by intent/difficulty | 105 | every PR (CI gate) |
| **Pipeline eval** (`runner.py`) | The whole crew: SQL, chart, narrative, critic, chaos robustness | 18 | nightly |
| **Spider dev / BIRD mini-dev** (`spider_eval.py`, `bird_eval.py`) | Public text-to-SQL benchmarks, comparable with published numbers | 1,034 / 500 (nightly: 200-question stratified subsets) | nightly + on demand |

Every accuracy number carries a 95% bootstrap CI; every A/B carries an
exact McNemar test. See [Methodology](#methodology).

## Why this exists

Multi-agent LLM systems are easy to demo, hard to *measure*. Without a
golden set you can't tell whether a prompt tweak helped or hurt, you
can't compare models, and you can't put a number in the README. This
harness gives every change a regression check.

## Layout

```
benchmarks/
├── text2sql.py          # shared harness: real coder path, official matching, cache, concurrency
├── stats.py             # bootstrap CI, paired Δ CI, exact McNemar, CI-aware gate
├── spider_eval.py       # Spider dev (+ spider_hardness.py: official difficulty port)
├── bird_eval.py         # BIRD mini-dev, evidence on/off
├── golden_sql.yaml      # 105 Chinook questions + gold SQL
├── golden_sql_eval.py   # runs them; golden_sql_check.py validates gold + ties
├── ablate.py            # ablations.yaml → comparison table with CIs + p-values
├── gate.py              # CI-aware regression gate; baselines/ holds committed baselines
├── dashboard.py         # static eval dashboard for GitHub Pages
├── error_sample.py      # sample failures for hand labeling (docs/error_taxonomy.md)
├── cases.yaml           # pipeline eval: 18 cases (12 golden + 6 adversarial chaos)
├── scorers.py           # pipeline eval scorers (7 rule-based + LLM judge, 3 chaos-only)
└── runner.py            # pipeline eval CLI
```

## Scoring dimensions

| Dimension | Weight | What it checks |
|---|---|---|
| `execution`           | 2.0 | Did the crew finish without error? |
| `sql_keywords`        | 1.0 | Does the SQL contain the expected columns/operators? |
| `top_value`           | 1.5 | Does row 0 of the result match the known ground truth? |
| `row_count`           | 0.5 | Is the result row count in the expected range? |
| `chart_type`          | 0.7 | Did the visualizer pick a sensible chart? |
| `narrative_keywords`  | 1.0 | Does the narrative mention the key entity? |
| `confidence`          | 0.5 | Did the Critic agree (≥ min confidence)? |
| `faithfulness` (LLM)  | 1.5 | Is every claim in the narrative supported by the result table? |

A case **passes** when every dimension passes its threshold.

## Running

```bash
make eval                                  # full run with LLM judge
make eval-fast                             # skip judge — pure rule-based, free
python -m benchmarks.runner --tag ranking  # filter by intent tag
python -m benchmarks.runner --case ec_top_channel
python -m benchmarks.runner --fail-under 0.75   # CI gate
```

Outputs land in `outputs/eval/`:

- `report.md` — human-readable per-case scorecard
- `summary.json` — headline metrics for CI / dashboards
- `results.json` — raw scorecards (one per case)

## Adding cases

1. Pick a dataset under `data/`.
2. Compute the ground truth offline (e.g. in a notebook with pandas).
3. Add a YAML entry to `cases.yaml` with `expected:` assertions.
4. Re-run `make eval` and confirm the new case passes.

Cases should be **deterministic** (same SQL → same answer) and cover
distinct intents — aggregation, ranking, time-series trend, ratio,
hypothesis comparison. Tag every case so per-intent regressions show up.

## CI integration

```yaml
# .github/workflows/eval.yml
- run: make eval-fast
- run: python -m benchmarks.runner --no-judge --fail-under 0.70
```

Use `--no-judge` in CI to keep runs deterministic and free; reserve the
LLM judge for nightly or pre-release runs.

## Interpreting the score

A weighted score in the **0.85+** range across 12+ cases is the bar for
"this agent crew is dependable on its golden set." Below 0.70 indicates
something regressed — check the per-dimension table to see whether the
hit was in SQL generation, narrative quality, or chart selection.

## Methodology

**Same code path as production.** The text-to-SQL benchmarks call
`coder_agent.generate_and_execute`: the production prompt, the
three-layer SQL guard and the self-repair loop. `--mode single` sets
`max_attempts=1` to measure what repair adds. `--oracle` feeds the gold
SQL through the same harness and must score 100% (Spider 1034/1034;
Chinook 105/105; BIRD 498/500, where two gold queries exceed BIRD's
official 30s timeout on a laptop). The oracle caught a real bug: pandas
turned SQL NULL into NaN, so any result with a NULL could never match.

**Official matching.** Spider uses the test-suite `result_eq`: same row
count and arity, column permutations allowed, order compared only when
the gold SQL has ORDER BY, rows compared as multisets. BIRD uses its
official set equality. Spider difficulty is a port of the official
`eval_hardness` and reproduces the published 248/446/174/166 split.

**Confidence intervals.** Accuracy is the mean of 0/1 outcomes, so the
bootstrap distribution is exactly Binomial(n, p̂)/n; `stats.py` samples
that directly. On n=200 a 95% CI is about ±6 points; on Spider's full
1,034 about ±2.8. A change smaller than that is not a finding.

**Paired comparisons.** Two configs on the same questions are compared
with the exact McNemar test: only questions where exactly one config is
right carry information, and under no difference each is a fair coin.
`ablate.py` reports Δ with a paired-bootstrap CI and the McNemar p.

**CI-aware gate.** `gate.py` fails a PR only when the new run's *upper*
95% bound is below the baseline's point estimate. A plain
`score < baseline` gate fails about half of all no-op changes on a noisy
eval; this one fails only on drops larger than noise. The cost is that
small real regressions pass on a small set; the fix is more cases, which
is why CI gates on the 105-question set, not the 18-case pipeline eval.

**No dev-set leakage.** Prompts are not tuned on Spider/BIRD dev
questions; any few-shot memory is built from train splits only.

**Caching.** Per-example results are cached under `outputs/cache/` keyed
by (benchmark, model, mode, evidence, flags, coder-prompt SHA), so an
ablation rerun pays only for new configs and a prompt edit invalidates
exactly what it should. `python -m benchmarks.ablate --plan` shows the
uncached work before spending anything.

