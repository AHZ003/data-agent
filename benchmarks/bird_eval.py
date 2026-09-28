"""BIRD mini-dev — execution accuracy of the real DataAgent coder.

BIRD is harder and closer to real analytics than Spider: wider schemas,
messy values, and questions that need domain knowledge. Each question
ships an "evidence" hint (e.g. "eligible free rate = Free Meal Count /
Enrollment"). Published numbers are reported both with and without it,
so we support both: --evidence on|off.

Usage:
    bash benchmarks/bird_setup.sh                 # one-time, ~800 MB
    python -m benchmarks.bird_eval --evidence on
    python -m benchmarks.bird_eval --evidence off --subset 200
    python -m benchmarks.bird_eval --oracle        # harness self-check, no LLM

Scoring uses the official BIRD rule: set(pred_rows) == set(gold_rows).
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

from benchmarks.text2sql import (  # noqa: E402
    Example,
    RunConfig,
    run_examples,
    stratified_subset,
    summarize,
    write_outputs,
)

BIRD_DIR = ROOT / "benchmarks" / "bird"
CACHE_DIR = ROOT / "outputs" / "cache"


def load_examples(bird_dir: Path = BIRD_DIR, db_id: Optional[str] = None,
                  difficulty: Optional[str] = None) -> list[Example]:
    path = bird_dir / "mini_dev_sqlite.json"
    if not path.exists():
        print("BIRD mini-dev not found. Run: bash benchmarks/bird_setup.sh", file=sys.stderr)
        sys.exit(2)
    examples = [
        Example(
            id=f"bird-minidev-{d['question_id']}",
            db_id=d["db_id"],
            question=d["question"],
            gold_sql=d["SQL"],
            difficulty=d.get("difficulty", ""),
            evidence=d.get("evidence", "") or "",
        )
        for d in json.loads(path.read_text())
    ]
    if db_id:
        examples = [e for e in examples if e.db_id == db_id]
    if difficulty:
        examples = [e for e in examples if e.difficulty == difficulty]
    return examples


def db_path_for(ex: Example, bird_dir: Path = BIRD_DIR) -> Path:
    return bird_dir / "dev_databases" / ex.db_id / f"{ex.db_id}.sqlite"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BIRD mini-dev execution accuracy")
    parser.add_argument("--evidence", choices=["on", "off"], default="on")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--subset", type=int, help="stratified random subset (by difficulty)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--db-id")
    parser.add_argument("--difficulty", choices=["simple", "moderate", "challenging"])
    parser.add_argument("--mode", choices=["single", "repair"], default="repair")
    parser.add_argument("--model")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--oracle", action="store_true",
                        help="feed gold SQL through the harness instead of the LLM")
    args = parser.parse_args(argv)

    from config import MODEL_NAME

    os.environ.setdefault("DATAAGENT_LLM_BACKOFF_RETRIES", "5")
    examples = load_examples(db_id=args.db_id, difficulty=args.difficulty)
    if args.subset:
        examples = stratified_subset(examples, args.subset, args.seed)
    elif args.limit:
        examples = examples[: args.limit]

    if args.oracle:
        return _oracle(examples)

    cfg = RunConfig(benchmark="bird", model=args.model or MODEL_NAME, mode=args.mode,
                    evidence=args.evidence == "on")
    results = run_examples(
        examples, db_path_for, cfg, rule="bird",
        concurrency=args.concurrency, cache_dir=None if args.no_cache else CACHE_DIR,
    )
    summary = summarize(results, cfg)
    out = args.out or f"outputs/bird/evidence_{args.evidence}"
    write_outputs(ROOT / out, f"DataAgent — BIRD mini-dev (evidence {args.evidence})", results, summary)
    lo, hi = summary["ci"]
    print(f"\nEX {summary['ex']:.1%} (95% CI {lo:.1%}–{hi:.1%}), n={summary['n']}, "
          f"${summary['cost_usd_total']:.4f}. Report: {out}/report.md")
    return 0


def _oracle(examples: list[Example]) -> int:
    from unittest.mock import patch

    from agents import coder_agent

    gold = {ex.question: ex.gold_sql for ex in examples}
    cfg = RunConfig(benchmark="oracle", model="gold", mode="single")
    with patch.object(coder_agent, "_generate_sql", side_effect=lambda q, ds, **k: gold[q]):
        results = run_examples(examples, db_path_for, cfg, rule="bird", progress=False)
    bad = [r for r in results if not r.match]
    print(f"Oracle EX: {1 - len(bad) / len(results):.2%} ({len(results) - len(bad)}/{len(results)})")
    for r in bad[:10]:
        print(f"  MISS {r.id} [{r.db_id}] {(r.error or r.gold_error or 'result mismatch')[:160]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
