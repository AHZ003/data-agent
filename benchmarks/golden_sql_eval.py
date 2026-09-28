"""Internal multi-table golden set (Chinook) — execution accuracy.

    python -m benchmarks.golden_sql_eval                # repair mode
    python -m benchmarks.golden_sql_eval --mode single
    python -m benchmarks.golden_sql_eval --oracle       # harness self-check, no LLM

105 hand-written questions with gold SQL over the 11-table Chinook
database (benchmarks/golden_sql.yaml), tagged by intent and difficulty,
run through the same production coder path and harness as Spider.
Unlike the 18-case pipeline eval (benchmarks/runner.py), this scores
only the SQL answer, which makes it cheap enough for every PR.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.golden_sql_check import load  # noqa: E402
from benchmarks.stats import bootstrap_ci  # noqa: E402
from benchmarks.text2sql import (  # noqa: E402
    Example, RunConfig, render_report, run_examples, summarize, write_outputs,
)

CACHE_DIR = ROOT / "outputs" / "cache"


def load_examples(include_user: bool = True) -> tuple[Path, list[Example], dict[str, list[str]]]:
    """Chinook golden cases, plus cases promoted from user feedback (golden_user.yaml)."""
    db, cases = load()
    user = ROOT / "benchmarks" / "golden_user.yaml"
    if include_user and user.exists():
        _, extra = load(user)
        cases = cases + extra
    examples = [Example(c["id"], "chinook", c["question"], c["gold_sql"], c["difficulty"]) for c in cases]
    return db, examples, {c["id"]: c.get("tags", []) for c in cases}


def by_tag(results, tags: dict[str, list[str]]) -> list[str]:
    groups: dict[str, list[bool]] = {}
    for r in results:
        for t in tags.get(r.id, []):
            groups.setdefault(t, []).append(r.match)
    lines = ["", "## By intent tag", "", "| Tag | n | EX | 95% CI |", "|---|---|---|---|"]
    for t, m in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ci = bootstrap_ci(m)
        lines.append(f"| {t} | {len(m)} | {ci.point:.1%} | {ci.lo:.1%}–{ci.hi:.1%} |")
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Chinook golden SQL set")
    parser.add_argument("--mode", choices=["single", "repair"], default="repair")
    parser.add_argument("--model")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", default="outputs/golden_sql")
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args(argv)

    from config import MODEL_NAME

    os.environ.setdefault("DATAAGENT_LLM_BACKOFF_RETRIES", "5")
    db, examples, tags = load_examples()
    if args.oracle:
        from unittest.mock import patch
        from agents import coder_agent
        gold = {ex.question: ex.gold_sql for ex in examples}
        cfg = RunConfig(benchmark="oracle", model="gold", mode="single")
        with patch.object(coder_agent, "_generate_sql", side_effect=lambda q, ds, **k: gold[q]):
            results = run_examples(examples, lambda ex: db, cfg, rule="spider", progress=False)
        bad = [r for r in results if not r.match]
        print(f"Oracle EX: {len(results) - len(bad)}/{len(results)}")
        for r in bad:
            print(f"  MISS {r.id}: {r.error or 'result mismatch'}")
        return 1 if bad else 0

    cfg = RunConfig(benchmark="golden_sql", model=args.model or MODEL_NAME, mode=args.mode)
    results = run_examples(examples, lambda ex: db, cfg, rule="spider",
                           concurrency=args.concurrency, cache_dir=None if args.no_cache else CACHE_DIR)
    summary = summarize(results, cfg)
    out = ROOT / args.out
    write_outputs(out, "DataAgent — Chinook golden SQL set", results, summary)
    report = out / "report.md"
    report.write_text(report.read_text() + "\n".join(by_tag(results, tags)) + "\n")
    lo, hi = summary["ci"]
    print(f"\nEX {summary['ex']:.1%} (95% CI {lo:.1%}–{hi:.1%}), n={summary['n']}. Report: {args.out}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
