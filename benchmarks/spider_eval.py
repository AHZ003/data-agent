"""Spider 1.0 dev — execution accuracy of the real DataAgent coder.

Runs Spider dev questions through `coder_agent.generate_and_execute`
(production prompt, three-layer guard, self-repair) against each
read-only database and scores with the official test-suite result
equivalence. Difficulty labels come from a port of the official
`eval_hardness` (benchmarks/spider_hardness.py).

Usage:
    bash benchmarks/spider_setup.sh                       # one-time, ~95 MB
    python -m benchmarks.spider_eval --limit 50
    python -m benchmarks.spider_eval --subset 200 --seed 0   # stratified by difficulty
    python -m benchmarks.spider_eval --mode single           # no self-repair
    python -m benchmarks.spider_eval --difficulty hard --concurrency 4
    python -m benchmarks.spider_eval --model gemini-2.5-pro
    python -m benchmarks.spider_eval --oracle                # harness self-check, no LLM

Outputs (default outputs/spider/): results.json, summary.json, report.md.
Per-example results are cached under outputs/cache/spider/<config>.jsonl,
so reruns of the same config are free and interrupted runs resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.spider_hardness import hardness  # noqa: E402
from benchmarks.text2sql import (  # noqa: E402
    Example,
    RunConfig,
    run_examples,
    stratified_subset,
    summarize,
    write_outputs,
)

SPIDER_DIR = ROOT / "benchmarks" / "spider"
CACHE_DIR = ROOT / "outputs" / "cache"


def load_examples(
    spider_dir: Path = SPIDER_DIR,
    db_id: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> list[Example]:
    path = spider_dir / "dev.json"
    if not path.exists():
        print("Spider dev set not found. Run: bash benchmarks/spider_setup.sh", file=sys.stderr)
        sys.exit(2)
    raw = json.loads(path.read_text())
    examples = [
        Example(
            id=f"spider-dev-{i}",
            db_id=d["db_id"],
            question=d["question"],
            gold_sql=d["query"],
            difficulty=hardness(d["sql"]),
        )
        for i, d in enumerate(raw)
    ]
    if db_id:
        examples = [e for e in examples if e.db_id == db_id]
    if difficulty:
        examples = [e for e in examples if e.difficulty == difficulty.lower()]
    return examples


def db_path_for(ex: Example, spider_dir: Path = SPIDER_DIR) -> Path:
    return spider_dir / "database" / ex.db_id / f"{ex.db_id}.sqlite"


def select(examples: list[Example], limit: Optional[int], subset: Optional[int], seed: int) -> list[Example]:
    if subset:
        return stratified_subset(examples, subset, seed)
    return examples[:limit] if limit else examples


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Spider dev execution accuracy")
    parser.add_argument("--limit", type=int, help="first N examples")
    parser.add_argument("--subset", type=int, help="stratified random subset of N (by difficulty)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--db-id")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "extra"])
    parser.add_argument("--mode", choices=["single", "repair"], default="repair")
    parser.add_argument("--model", help="override DATAAGENT_MODEL")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", default="outputs/spider")
    parser.add_argument("--oracle", action="store_true",
                        help="feed gold SQL through the harness instead of the LLM; must score 100%%")
    args = parser.parse_args(argv)

    from config import MODEL_NAME

    os.environ.setdefault("DATAAGENT_LLM_BACKOFF_RETRIES", "5")
    cfg = RunConfig(benchmark="spider", model=args.model or MODEL_NAME, mode=args.mode)
    examples = select(load_examples(db_id=args.db_id, difficulty=args.difficulty), args.limit, args.subset, args.seed)
    if not examples:
        print("No examples matched the filters.", file=sys.stderr)
        return 2

    if args.oracle:
        return _oracle(examples)

    results = run_examples(
        examples, db_path_for, cfg, rule="spider",
        concurrency=args.concurrency, cache_dir=None if args.no_cache else CACHE_DIR,
    )
    summary = summarize(results, cfg)
    write_outputs(ROOT / args.out, "DataAgent — Spider 1.0 dev", results, summary)
    lo, hi = summary["ci"]
    print(f"\nEX {summary['ex']:.1%} (95% CI {lo:.1%}–{hi:.1%}), n={summary['n']}, "
          f"${summary['cost_usd_total']:.4f}. Report: {args.out}/report.md")
    return 0


def _oracle(examples: list[Example]) -> int:
    """Self-check: gold SQL through guard, execution and matching must score 100%."""
    from unittest.mock import patch

    from agents import coder_agent

    gold = {ex.question: ex.gold_sql for ex in examples}
    cfg = RunConfig(benchmark="oracle", model="gold", mode="single")
    with patch.object(coder_agent, "_generate_sql", side_effect=lambda q, ds, **k: gold[q]):
        results = run_examples(examples, db_path_for, cfg, rule="spider", progress=False)
    bad = [r for r in results if not r.match]
    print(f"Oracle EX: {1 - len(bad) / len(results):.2%} ({len(results) - len(bad)}/{len(results)})")
    for r in bad[:10]:
        print(f"  MISS {r.id} [{r.db_id}] {r.error or 'result mismatch'}: {r.gold_sql[:100]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
