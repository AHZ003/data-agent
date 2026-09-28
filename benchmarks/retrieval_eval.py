"""Schema-linking recall@k — measured separately from generation.

    python -m benchmarks.retrieval_eval --benchmark spider
    python -m benchmarks.retrieval_eval --benchmark bird --embedder hashing --k 10 20 30

For each question: parse the gold SQL, find the tables/columns it uses,
run the schema linker, and check whether all of them survived pruning.
Reports per k: column recall (share of gold columns kept), strict recall
(all gold columns kept), table recall, and prompt-size reduction. With
the hashing embedder this needs no API key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.stats import bootstrap_ci  # noqa: E402


def evaluate(examples, db_path_for, ks: list[int], embedder, min_columns_to_prune: int = 0,
             values: bool = False) -> dict:
    from core.datasource import DataSource
    from retrieval.gold_schema import gold_items
    from retrieval.schema_linking import SchemaLinker, prune

    import sqlite3

    from retrieval.values import build

    linkers: dict[Path, SchemaLinker] = {}
    value_indexes: dict[Path, object] = {}
    per_k = {k: {"col_recall": [], "strict": [], "table_recall": [], "size_ratio": []} for k in ks}
    skipped = 0
    for ex in examples:
        path = db_path_for(ex)
        if path not in linkers:
            linkers[path] = SchemaLinker(DataSource.from_sqlite(str(path), name=ex.db_id), embedder)
            if values:
                conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                conn.text_factory = lambda b: b.decode(errors="replace")
                value_indexes[path] = build(conn, linkers[path].ds)
        linker = linkers[path]
        g_tables, g_cols = gold_items(ex.gold_sql, linker.ds)
        if not g_cols:
            skipped += 1  # e.g. SELECT count(*) FROM t: no columns to recall
            continue
        query = ex.question if not ex.evidence else f"{ex.question} {ex.evidence}"
        ranked = linker.rank(query)
        must_keep = {(m.table, m.column) for m in value_indexes[path].match(query)} if values else set()
        full = len(linker.ds.to_prompt())
        for k in ks:
            if len(linker.keys) <= min_columns_to_prune:
                pruned = linker.ds
            else:
                pruned = prune(linker.ds, set(ranked[:k]) | must_keep)
            kept = {(t.name.lower(), c.name.lower()) for t in pruned.tables for c in t.columns}
            kept_t = {t.name.lower() for t in pruned.tables}
            hit = len(g_cols & kept) / len(g_cols)
            per_k[k]["col_recall"].append(hit)
            per_k[k]["strict"].append(hit == 1.0)
            per_k[k]["table_recall"].append(len(g_tables & kept_t) / max(len(g_tables), 1))
            per_k[k]["size_ratio"].append(len(pruned.to_prompt()) / full)
    name = getattr(embedder, "name", "bm25-only") + (" + value hints" if values else "")
    out = {"n": len(examples) - skipped, "skipped_no_columns": skipped, "embedder": name, "k": {}}
    for k, m in per_k.items():
        strict = bootstrap_ci(m["strict"])
        n = max(len(m["col_recall"]), 1)
        out["k"][k] = {
            "column_recall": sum(m["col_recall"]) / n,
            "strict_recall": strict.point, "strict_ci": [strict.lo, strict.hi],
            "table_recall": sum(m["table_recall"]) / n,
            "prompt_size_ratio": sum(m["size_ratio"]) / n,
        }
    return out


def render(bench: str, res: dict) -> str:
    lines = [f"### {bench} — schema linking ({res['embedder']}, n={res['n']})", "",
             "| k | column recall | all gold columns kept (95% CI) | table recall | prompt size vs full |",
             "|---|---|---|---|---|"]
    for k, m in res["k"].items():
        lines.append(f"| {k} | {m['column_recall']:.1%} | {m['strict_recall']:.1%} "
                     f"({m['strict_ci'][0]:.1%}–{m['strict_ci'][1]:.1%}) | {m['table_recall']:.1%} | "
                     f"{m['prompt_size_ratio']:.0%} |")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["spider", "bird", "golden_sql"], default="spider")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20, 30])
    parser.add_argument("--embedder", choices=["none", "hashing", "gemini"], default="hashing")
    parser.add_argument("--values", action="store_true", help="value matches become must-keep columns")
    parser.add_argument("--out", default="outputs/retrieval")
    args = parser.parse_args(argv)

    from retrieval.embed import GeminiEmbedder, HashingEmbedder
    embedder = {"none": None, "hashing": HashingEmbedder(), "gemini": None}[args.embedder]
    if args.embedder == "gemini":
        embedder = GeminiEmbedder()

    if args.benchmark == "spider":
        from benchmarks import spider_eval as mod
        examples, db_path_for = mod.load_examples(), mod.db_path_for
    elif args.benchmark == "bird":
        from benchmarks import bird_eval as mod
        examples, db_path_for = mod.load_examples(), mod.db_path_for
    else:
        from benchmarks.golden_sql_eval import load_examples
        db, examples, _ = load_examples()
        db_path_for = lambda ex: db  # noqa: E731

    res = evaluate(examples, db_path_for, args.k, embedder, values=args.values)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.benchmark}_{args.embedder}" + ("_values" if args.values else "")
    (out / f"{stem}.json").write_text(json.dumps(res, indent=2))
    md = render(args.benchmark, res)
    (out / f"{stem}.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
