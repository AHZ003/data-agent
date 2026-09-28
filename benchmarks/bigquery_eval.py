"""BigQuery golden set: accuracy and bytes scanned, with/without partition hints.

    uv sync --extra bigquery && gcloud auth application-default login
    python -m benchmarks.bigquery_eval --check            # dry-run every gold query (free)
    python -m benchmarks.bigquery_eval                    # full run, partition hints on
    python -m benchmarks.bigquery_eval --no-partition-hints

Reports EX (Spider matching rule), bytes each predicted query scans (from
a free dry run), and how often the cost guard fired. Comparing the two
hint settings measures what partition-aware prompting saves.

Gold results are cached in outputs/cache/bigquery_gold.json so gold
queries are billed once.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES = ROOT / "benchmarks" / "cases_bigquery.yaml"
GOLD_CACHE = ROOT / "outputs" / "cache" / "bigquery_gold.json"


def load():
    data = yaml.safe_load(CASES.read_text())
    return data["tables"], data["cases"]


def _norm(rows) -> list[tuple]:
    """Gold is cached as JSON: compare numbers as numbers, everything else as text."""
    return [tuple(v if v is None or isinstance(v, (int, float)) else str(v) for v in r) for r in rows]


def strip_partition_hints(ds):
    tables = []
    for t in ds.tables:
        cols = [replace(c, description=None) if c.description and "PARTITION" in c.description.upper()
                or (c.description and "clustering" in c.description) else c for c in t.columns]
        tables.append(replace(t, columns=cols))
    return replace(ds, tables=tables)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="dry-run the gold SQL only")
    parser.add_argument("--no-partition-hints", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args(argv)

    from agents.coder_agent import generate_and_execute
    from benchmarks.stats import bootstrap_ci
    from benchmarks.text2sql import df_to_rows, results_match
    from core.bigquery_engine import BigQueryEngine

    tables, cases = load()
    eng = BigQueryEngine(tables, name="nyc_taxi")

    if args.check:
        bad = 0
        for c in cases:
            try:
                est = eng.dry_run(c["gold_sql"])
                print(f"ok   {c['id']}  {est.fmt()}")
            except Exception as e:
                bad += 1
                print(f"FAIL {c['id']}  {str(e).splitlines()[0][:160]}")
        return 1 if bad else 0

    gold = json.loads(GOLD_CACHE.read_text()) if GOLD_CACHE.exists() else {}
    if args.no_partition_hints:
        eng._datasource = strip_partition_hints(eng.datasource())

    rows, matches = [], []
    for c in cases:
        if c["id"] not in gold:
            df, err = eng.execute_query(c["gold_sql"], confirm_cost=True)
            if err:
                print(f"gold failed for {c['id']}: {err}")
                continue
            gold[c["id"]] = [list(r) for r in df_to_rows(df)]
            GOLD_CACHE.parent.mkdir(parents=True, exist_ok=True)
            GOLD_CACHE.write_text(json.dumps(gold, default=str))
        df, res = generate_and_execute(c["question"], eng, model=args.model)
        scanned = eng.dry_run(res.sql_query).bytes_processed if res.sql_query else None
        ok = df is not None and results_match("spider", _norm(gold[c["id"]]), _norm(df_to_rows(df)), c["gold_sql"])
        matches.append(ok)
        rows.append({"id": c["id"], "match": ok, "bytes": scanned, "error": res.error, "sql": res.sql_query})
        print(f"{c['id']} {'MATCH' if ok else 'miss '} {scanned / 2**30 if scanned else 0:8.2f} GiB  {res.error or ''}")

    ci = bootstrap_ci(matches)
    scanned = [r["bytes"] for r in rows if r["bytes"]]
    guard = sum(1 for r in rows if r["error"] and ("CostGuard" in r["error"] or "CostConfirmation" in r["error"]))
    label = "no partition hints" if args.no_partition_hints else "partition hints"
    print(f"\n{label}: EX {ci.fmt()} n={len(matches)}; mean scanned "
          f"{(sum(scanned) / max(len(scanned), 1)) / 2**30:.2f} GiB; cost guard fired {guard}x")
    out = ROOT / "outputs" / "bigquery"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_{'nohints' if args.no_partition_hints else 'hints'}.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
