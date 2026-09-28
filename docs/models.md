# Models and the cost/accuracy frontier

Question: for text-to-SQL on this stack, what does each point of execution
accuracy cost, and can routing or a fine-tuned small model get close to the
strong model's accuracy for less?

## Configurations

All run through the same harness (`benchmarks/text2sql.py`), same prompts,
same guard, same self-repair loop, on the same fixed 200-question stratified
Spider dev subset (`benchmarks/ablations.yaml`, section P7):

| Config | Model | Notes |
|---|---|---|
| `spider-flash-repair` | Gemini 2.5 Flash | default |
| `spider-pro-repair` | Gemini 2.5 Pro | strong model |
| `spider-qwen7b-base` | Qwen2.5-Coder 7B via Ollama | open model, no fine-tuning |
| `spider-finetuned-1.5b` | Qwen2.5-Coder 1.5B + QLoRA | `training/`, Spider train only |
| `spider-router` | Flash, escalating to Pro | `routing/`, trained on held-out half |

Any agent can use any model: `DATAAGENT_MODEL_<AGENT>=ollama/<model>` or
`gemini-*` (`core/llm.py`).

## How cost is counted

- Gemini: tokens × list price (`core/tracing.MODEL_PRICING`), including every
  self-repair attempt.
- Local models: no API bill; cost = generation seconds × `OLLAMA_USD_PER_HOUR`,
  the hourly price of the hardware used. The rate must be stated with results.

## Method

- Accuracy with 95% bootstrap CIs; every model vs the Flash baseline with a
  paired Δ CI and exact McNemar p (`python -m benchmarks.ablate`).
- Router: fit on half the questions, evaluated on the other half by replaying
  cached per-question outcomes of both models across thresholds; the chosen
  threshold is the cheapest within 1 point of Pro (`python -m routing.train`).
- Frontier: `python -m benchmarks.pareto --benchmark spider` →
  `docs/pareto_spider.svg`.

## Results

**Pending.** Needs, in order:

1. A valid `GOOGLE_API_KEY`, then
   `python -m benchmarks.ablate --only spider-flash-repair,spider-pro-repair`.
2. `ollama pull qwen2.5-coder:7b`, then `--only spider-qwen7b-base`.
3. The QLoRA run on a GPU (`training/qlora_unsloth.ipynb`), `ollama create
   dataagent-sql`, then `--only spider-finetuned-1.5b`.
4. `python -m routing.train --cheap spider-flash-repair --strong spider-pro-repair`,
   then `--only spider-router`.
5. `python -m benchmarks.pareto --benchmark spider`.

Expected shape, to be confirmed or refuted: the fine-tuned 1.5B model will not
beat Pro; the interesting numbers are where it lands per dollar and whether the
router recovers most of Pro's accuracy at a fraction of its cost.
