"""BigQuery golden set: accuracy and bytes scanned, with/without partition hints.

    uv sync --extra bigquery && gcloud auth application-default login
    python scripts/bigquery_setup.py --project $BQ_PROJECT   # once: partitioned copy
    python -m benchmarks.bigquery_eval --check            # dry-run every gold query (free)
    python -m benchmarks.bigquery_eval                    # full run, partition hints on
    python -m benchmarks.bigquery_eval --no-partition-hints

Reports EX (Spider matching rule), bytes each predicted query scans (from
a free dry run), and how often the cost guard fired. Two EX numbers:
strict, and "with extra columns", where the gold columns must appear in the
prediction but more may follow. The coder is told to return the measure it
ranked by ("which hour" -> hour and count), which strict matching counts
as wrong. Comparing the two
hint settings measures what partition-aware prompting saves.

Gold results are cached in outputs/cache/bigquery_gold.json so gold
queries are billed once. Bytes are reported separately for questions with a
date range (`pruned`: a partition filter can cut the scan) and whole-year
questions (nothing to prune).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from dataclasses import replace
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES = ROOT / "benchmarks" / "cases_bigquery.yaml"
GOLD_CACHE = ROOT / "outputs" / "cache" / "bigquery_gold.json"
PROJECT = os.getenv("BQ_PROJECT", "gen-lang-client-0767549601")


def load(project: str = PROJECT):
    data = yaml.safe_load(CASES.read_text())
    tables = [t.format(project=project) for t in data["tables"]]
    cases = [{**c, "gold_sql": c["gold_sql"].format(project=project)} for c in data["cases"]]
    return tables, cases


def _value(v):
    """BigQuery NUMERIC comes back as Decimal and a model may CAST to FLOAT64:
    compare all numbers as floats rounded to 4 places, everything else as text."""
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float, Decimal)):
        f = round(float(v), 4)
        return int(f) if f.is_integer() else f
    try:
        return _value(Decimal(str(v)))  # numbers cached as JSON strings
    except Exception:
        return str(v)


def _norm(rows) -> list[tuple]:
    return [tuple(_value(v) for v in r) for r in rows]


def match_extra_columns(gold: list, pred: list, gold_sql: str) -> bool:
    """True if some choice of len(gold[0]) prediction columns matches gold."""
    from itertools import permutations
    from benchmarks.text2sql import results_match
    if not gold or not pred or len(pred[0]) < len(gold[0]):
        return results_match("spider", gold, pred, gold_sql)
    k = len(gold[0])
    for cols in permutations(range(len(pred[0])), k):
        if results_match("spider", gold, [tuple(r[i] for i in cols) for r in pred], gold_sql):
            return True
    return False


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
    parser.add_argument("--resume", action="store_true", help="skip cases already in the partial results file")
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

    tag = "nohints" if args.no_partition_hints else "hints"
    out = ROOT / "outputs" / "bigquery"
    out.mkdir(parents=True, exist_ok=True)
    partial = out / f"partial_{tag}.json"
    done = {r["id"]: r for r in json.loads(partial.read_text())} if args.resume and partial.exists() else {}

    rows, matches, matches_extra = [], [], []
    for c in cases:
        if c["id"] in done:  # finished in an earlier, interrupted run
            r = done[c["id"]]
            rows.append(r)
            matches.append(r["match"])
            matches_extra.append(r["match_extra_cols"])
            continue
        if c["id"] not in gold:
            df, err = eng.execute_query(c["gold_sql"], confirm_cost=True)
            if err:
                print(f"gold failed for {c['id']}: {err}")
                continue
            gold[c["id"]] = [list(r) for r in _norm(df_to_rows(df))]
            GOLD_CACHE.parent.mkdir(parents=True, exist_ok=True)
            GOLD_CACHE.write_text(json.dumps(gold, default=str))
        df, res = generate_and_execute(c["question"], eng, model=args.model)
        scanned = eng.dry_run(res.sql_query).bytes_processed if res.sql_query else None
        g, p = _norm(gold[c["id"]]), (_norm(df_to_rows(df)) if df is not None else None)
        ok = p is not None and results_match("spider", g, p, c["gold_sql"])
        ok_extra = p is not None and (ok or match_extra_columns(g, p, c["gold_sql"]))
        matches.append(ok)
        matches_extra.append(ok_extra)
        rows.append({"id": c["id"], "match": ok, "match_extra_cols": ok_extra, "bytes": scanned, "error": res.error, "sql": res.sql_query,
                     "tags": c.get("tags", [])})
        partial.write_text(json.dumps(rows, indent=2, default=str))  # survive an interrupted run
        print(f"{c['id']} {'MATCH' if ok else ('extra' if ok_extra else 'miss ')} {scanned / 2**30 if scanned else 0:8.2f} GiB  {res.error or ''}", flush=True)

    ci = bootstrap_ci(matches)
    ci_extra = bootstrap_ci(matches_extra)
    guard = sum(1 for r in rows if r["error"] and ("CostGuard" in r["error"] or "CostConfirmation" in r["error"]))

    def mean_gib(subset):
        b = [r["bytes"] for r in subset if r["bytes"]]
        return round(sum(b) / len(b) / 2**30, 4) if b else None

    pruned = [r for r in rows if "pruned" in r["tags"]]
    summary = {
        "hints": not args.no_partition_hints, "n": len(matches), "ex": ci.point, "ci": [ci.lo, ci.hi],
        "ex_extra_cols": ci_extra.point, "ci_extra_cols": [ci_extra.lo, ci_extra.hi],
        "mean_gib": mean_gib(rows), "mean_gib_pruned": mean_gib(pruned),
        "mean_gib_full_year": mean_gib([r for r in rows if "full_year" in r["tags"]]),
        "total_gib": round(sum(r["bytes"] or 0 for r in rows) / 2**30, 3), "cost_guard_fired": guard,
    }
    label = "no partition hints" if args.no_partition_hints else "partition hints"
    print(f"\n{label}: EX {ci.fmt()} strict, {ci_extra.fmt()} with extra columns, n={len(matches)}; mean scanned {summary['mean_gib']} GiB "
          f"(date-range questions {summary['mean_gib_pruned']} GiB); cost guard fired {guard}x")
    (out / f"results_{tag}.json").write_text(json.dumps(rows, indent=2, default=str))
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
