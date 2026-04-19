# DataAgent — Evaluation Harness

A reproducible scorecard for the multi-agent crew. Runs a fixed set of
golden questions against the live pipeline and grades each answer on
seven dimensions plus an optional LLM-as-judge faithfulness check.

## Why this exists

Multi-agent LLM systems are easy to demo, hard to *measure*. Without a
golden set you can't tell whether a prompt tweak helped or hurt, you
can't compare models, and you can't put a number in the README. This
harness gives every change a regression check.

## Layout

```
benchmarks/
├── cases.yaml      # 12 golden (dataset, question, expected) tuples
├── scorers.py      # 7 scoring dimensions + LLM-as-judge
├── runner.py       # CLI: load cases → run crew → aggregate → report
└── README.md       # this file
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
