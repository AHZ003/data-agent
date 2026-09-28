"""Sample benchmark failures for hand-labeling, then summarize the labels.

    # 1. after a run, sample 50 misses into a CSV (category column blank):
    python -m benchmarks.error_sample sample outputs/spider/results.json \
        --out docs/error_analysis/spider_labels.csv
    # 2. fill in `category` (and `notes`) by hand, using docs/error_taxonomy.md
    # 3. summarize:
    python -m benchmarks.error_sample summarize docs/error_analysis/spider_labels.csv

The sample is deterministic (seeded) and stratified by difficulty so the
labeled set reflects where failures actually happen.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

CATEGORIES = (
    "schema_linking",      # wrong table/column chosen, or needed one missing
    "join",                # wrong join path, missing/extra join, wrong join key
    "aggregation",         # wrong aggregate, GROUP BY, HAVING, DISTINCT
    "value_mismatch",      # literal doesn't match stored value ('CA' vs 'California')
    "condition",           # wrong filter logic / comparison / NULL handling
    "ordering_limit",      # wrong ORDER BY direction, LIMIT, tie handling
    "output_shape",        # right logic but extra/missing/reordered columns
    "ambiguous_question",  # the question admits the predicted reading
    "questionable_gold",   # gold SQL is wrong or arguably wrong
    "execution_error",     # no executable SQL after repair
    "other",
)
FIELDS = ["id", "db_id", "difficulty", "question", "gold_sql", "pred_sql", "error", "category", "notes"]


def sample(results_path: Path, n: int = 50, seed: int = 0) -> list[dict]:
    misses = [r for r in json.loads(results_path.read_text()) if not r["match"]]
    rng = random.Random(seed)
    by_diff: dict[str, list[dict]] = {}
    for r in misses:
        by_diff.setdefault(r.get("difficulty", ""), []).append(r)
    picked = []
    for diff, group in sorted(by_diff.items()):
        k = max(1, round(n * len(group) / max(len(misses), 1)))
        picked += rng.sample(group, min(k, len(group)))
    rng.shuffle(picked)
    return picked[:n]


def summarize(labels_path: Path) -> str:
    rows = list(csv.DictReader(labels_path.open()))
    labeled = [r for r in rows if r["category"].strip()]
    unknown = sorted({r["category"] for r in labeled} - set(CATEGORIES))
    counts = Counter(r["category"].strip() for r in labeled)
    lines = [f"Labeled {len(labeled)}/{len(rows)} sampled failures.", "",
             "| Category | n | share |", "|---|---|---|"]
    for cat, c in counts.most_common():
        lines.append(f"| {cat} | {c} | {c / max(len(labeled), 1):.0%} |")
    if unknown:
        lines += ["", f"Unknown categories (fix or add to the taxonomy): {', '.join(unknown)}"]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("results", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("-n", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("summarize")
    m.add_argument("labels", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "sample":
        if args.out.exists():
            print(f"{args.out} exists; refusing to overwrite hand labels.", file=sys.stderr)
            return 1
        rows = sample(args.results, args.n, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({**r, "category": "", "notes": ""})
        print(f"wrote {len(rows)} failures to {args.out}; label the `category` column")
    else:
        print(summarize(args.labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
