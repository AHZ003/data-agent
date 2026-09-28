# Fine-tuning a small open coder model

Goal: see where a fine-tuned 1.5B–7B open model lands on the cost/accuracy
frontier next to Gemini Flash and Pro — not to beat Pro.

1. `python -m training.build_dataset` → `training/data/{train,val}.jsonl`
   (8,413 / 246 examples; Spider train only, validation = 7 held-out databases;
   prompts are the Coder's exact inference prompts).
2. Run `training/qlora_unsloth.ipynb` on a free Colab/Kaggle T4 (QLoRA, r=16,
   1 epoch, loss on the SQL answer only). It saves a LoRA adapter and a
   `q4_k_m` GGUF, and can push both to the Hugging Face Hub.
3. Serve the GGUF with Ollama and evaluate through the same harness as every
   other model:
   `python -m benchmarks.spider_eval --model ollama/dataagent-sql --subset 200`.
4. Add it to `benchmarks/ablations.yaml` and plot the frontier:
   `python -m benchmarks.pareto`.

**Contamination rules.** Train on Spider train only. Evaluate on Spider dev
(disjoint databases) and BIRD mini-dev (different benchmark entirely). The
builder refuses if any train database appears in dev; query memory refuses
evaluation ids. Say this in any write-up.

**Cost accounting.** A local model has no API bill. Report its cost as
hardware time: set `OLLAMA_USD_PER_HOUR` to the hourly price of the machine
you ran on (e.g. a rented GPU's rate) and state it next to the number.
