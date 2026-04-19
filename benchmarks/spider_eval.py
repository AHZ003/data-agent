"""Spider benchmark adapter — execution-accuracy evaluation.

Runs a configurable slice of the Spider 1.0 dev set through our
coder agent and computes execution accuracy (EX): the fraction of
predicted SQL queries whose result set exactly matches the gold SQL
result set when executed on the same SQLite database.

Usage:
    # Setup first (one-time, downloads ~95 MB):
    bash benchmarks/spider_setup.sh

    # Run evaluation:
    python -m benchmarks.spider_eval                     # full dev set (1034)
    python -m benchmarks.spider_eval --limit 50          # first 50
    python -m benchmarks.spider_eval --db-id world_1     # single DB
    python -m benchmarks.spider_eval --difficulty easy    # easy only
    python -m benchmarks.spider_eval --model gemini-2.5-pro

Output:
    outputs/spider/
        results.json       per-example results
        summary.json       headline metrics
        report.md          human-readable report
        errors.json        failed examples for debugging

Why Spider?

Spider is the standard public benchmark for text-to-SQL systems.
Quoting a concrete EX score against Spider dev (n=1034) is the single
biggest credibility signal for a data-agent portfolio project. It
lets a recruiter compare you against published numbers from research
labs and enterprise products on a common scale.

Design decisions:

- We call `coder_agent._generate_sql` directly (not the full crew)
  because Spider evaluates SQL correctness, not narrative quality or
  chart selection. Running the full orchestrator would add latency
  and cost without changing the SQL score.

- We build a `SemanticSchema` from the Spider database's actual
  SQLite schema (via PRAGMA table_info), not from our schema_agent's
  profiler. This isolates the coder's SQL-writing ability from the
  profiler's column-role classification.

- Execution accuracy is computed by sorting both result sets and
  comparing them row-by-row. This matches the official Spider eval's
  "exec" mode, minus the test-suite augmentation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPIDER_DIR = ROOT / "benchmarks" / "spider"


@dataclass
class SpiderResult:
    idx: int
    db_id: str
    question: str
    gold_sql: str
    pred_sql: str
    match: bool
    error: Optional[str] = None
    duration_s: float = 0.0


@dataclass
class SpiderSummary:
    n_total: int
    n_correct: int
    n_error: int
    execution_accuracy: float
    avg_latency_s: float
    model: str
    timestamp: str
    by_difficulty: dict = field(default_factory=dict)


def _load_dev(limit: Optional[int] = None,
              db_id: Optional[str] = None,
              difficulty: Optional[str] = None) -> list[dict]:
    path = SPIDER_DIR / "dev.json"
    if not path.exists():
        print("Spider dev set not found. Run:", file=sys.stderr)
        print("  bash benchmarks/spider_setup.sh", file=sys.stderr)
        sys.exit(2)
    with open(path) as f:
        data = json.load(f)
    if db_id:
        data = [d for d in data if d["db_id"] == db_id]
    if difficulty:
        data = [d for d in data if d.get("difficulty", "").lower() == difficulty.lower()]
    if limit:
        data = data[:limit]
    return data


def _get_spider_schema(db_path: Path) -> dict[str, list[dict]]:
    """Extract table schemas from a Spider SQLite database."""
    conn = sqlite3.connect(str(db_path))
    tables: dict[str, list[dict]] = {}
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    for (tname,) in cursor.fetchall():
        cols = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
        tables[tname] = [
            {"name": c[1], "dtype": c[2], "pk": bool(c[5])} for c in cols
        ]
    conn.close()
    return tables


def _schema_to_prompt_context(tables: dict[str, list[dict]]) -> str:
    """Format multi-table schema as a string for the coder prompt."""
    parts = []
    for tname, cols in tables.items():
        col_strs = [f"  {c['name']} ({c['dtype']})" +
                    (" [PK]" if c["pk"] else "") for c in cols]
        parts.append(f"Table: {tname}\n" + "\n".join(col_strs))
    return "\n\n".join(parts)


def _build_spider_prompt(question: str, schema_text: str,
                         db_tables: list[str]) -> str:
    """Build a coder prompt adapted for Spider's multi-table schemas."""
    return f"""You are an expert SQL analyst. Given a database schema and a
natural language question, write a valid SQLite SQL query to answer it.

Rules:
- Use ONLY column names that exist in the schema
- Use double quotes for column names with spaces or special characters
- Use appropriate aggregations (SUM, AVG, COUNT, etc.)
- For date operations, use SQLite date functions
- Output ONLY the SQL query, no explanation
- Available tables: {', '.join(db_tables)}

DATABASE SCHEMA:
{schema_text}

USER QUESTION: {question}

Write the SQL query:"""


def _extract_sql(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    cleaned = text.strip()
    for prefix in ["Here is", "The SQL", "SQL:", "Query:"]:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip().lstrip(":")
    return cleaned.strip()


def _exec_sql(db_path: Path, sql: str) -> Optional[list[tuple]]:
    """Execute SQL against a Spider SQLite DB. Returns sorted rows or None."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA case_sensitive_like = OFF")
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return sorted([tuple(str(v) for v in r) for r in rows])
    except Exception:
        return None


def _results_match(gold_rows: Optional[list], pred_rows: Optional[list]) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return gold_rows == pred_rows


def _generate_sql_for_spider(question: str, schema_text: str,
                              db_tables: list[str]) -> str:
    """Call Gemini to generate SQL for a Spider question."""
    from google import genai
    from google.genai import types as genai_types
    from config import GOOGLE_API_KEY, MODEL_NAME

    client = genai.Client(api_key=GOOGLE_API_KEY)
    prompt = _build_spider_prompt(question, schema_text, db_tables)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=1024,
        ),
    )
    return _extract_sql(response.text)


def run_spider_eval(examples: list[dict]) -> list[SpiderResult]:
    results: list[SpiderResult] = []
    for i, ex in enumerate(examples):
        db_id = ex["db_id"]
        question = ex["question"]
        gold_sql = ex["query"]
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            results.append(SpiderResult(
                idx=i, db_id=db_id, question=question,
                gold_sql=gold_sql, pred_sql="",
                match=False, error=f"DB not found: {db_path}",
            ))
            continue

        tables = _get_spider_schema(db_path)
        schema_text = _schema_to_prompt_context(tables)
        db_tables = list(tables.keys())

        start = time.time()
        pred_sql = ""
        error = None
        try:
            pred_sql = _generate_sql_for_spider(question, schema_text, db_tables)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        gold_rows = _exec_sql(db_path, gold_sql)
        pred_rows = _exec_sql(db_path, pred_sql) if pred_sql and not error else None

        match = _results_match(gold_rows, pred_rows)
        duration = time.time() - start

        results.append(SpiderResult(
            idx=i, db_id=db_id, question=question,
            gold_sql=gold_sql, pred_sql=pred_sql,
            match=match, error=error, duration_s=round(duration, 2),
        ))

        status = "MATCH" if match else ("ERROR" if error else "MISS")
        print(f"  [{i+1}/{len(examples)}] {db_id:25s} {status:5s}  "
              f"({duration:.1f}s)  {question[:60]}")

    return results


def summarize(results: list[SpiderResult], model: str) -> SpiderSummary:
    n = len(results)
    correct = sum(1 for r in results if r.match)
    errors = sum(1 for r in results if r.error)
    avg_lat = sum(r.duration_s for r in results) / max(n, 1)
    return SpiderSummary(
        n_total=n,
        n_correct=correct,
        n_error=errors,
        execution_accuracy=round(correct / max(n, 1), 4),
        avg_latency_s=round(avg_lat, 2),
        model=model,
        timestamp=datetime.now(tz=__import__('datetime').timezone.utc).isoformat(),
    )


def render_report(results: list[SpiderResult], summary: SpiderSummary) -> str:
    lines = [
        "# DataAgent — Spider 1.0 Dev Evaluation",
        "",
        f"_Generated {summary.timestamp} | Model: `{summary.model}`_",
        "",
        "## Headline",
        "",
        f"- **Execution accuracy (EX):** `{summary.execution_accuracy:.1%}` "
        f"({summary.n_correct}/{summary.n_total})",
        f"- **Errors:** {summary.n_error}",
        f"- **Avg latency:** {summary.avg_latency_s:.2f}s per question",
        "",
        "## Per-database breakdown",
        "",
        "| db_id | total | correct | accuracy |",
        "|-------|-------|---------|----------|",
    ]

    by_db: dict[str, list[bool]] = {}
    for r in results:
        by_db.setdefault(r.db_id, []).append(r.match)
    for db_id in sorted(by_db, key=lambda k: -sum(by_db[k]) / len(by_db[k])):
        matches = by_db[db_id]
        acc = sum(matches) / len(matches)
        lines.append(
            f"| `{db_id}` | {len(matches)} | {sum(matches)} | {acc:.0%} |"
        )

    lines.extend([
        "",
        "## Errors",
        "",
    ])
    error_results = [r for r in results if r.error]
    if error_results:
        for r in error_results[:20]:
            lines.append(f"- **[{r.db_id}]** {r.question[:80]}")
            lines.append(f"  `{r.error[:150]}`")
    else:
        lines.append("_No errors._")

    lines.extend([
        "",
        "## Sample misses (first 10)",
        "",
    ])
    misses = [r for r in results if not r.match and not r.error][:10]
    for r in misses:
        lines.append(f"### [{r.db_id}] {r.question}")
        lines.append(f"**Gold:** `{r.gold_sql[:200]}`")
        lines.append(f"**Pred:** `{r.pred_sql[:200]}`")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run Spider text-to-SQL eval")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples to evaluate")
    parser.add_argument("--db-id", default=None,
                        help="Run only examples for this database")
    parser.add_argument("--difficulty", default=None,
                        help="Filter by difficulty (easy/medium/hard/extra)")
    parser.add_argument("--model", default=None,
                        help="Override DATAAGENT_MODEL")
    parser.add_argument("--out", default="outputs/spider",
                        help="Output directory")
    args = parser.parse_args()

    if args.model:
        os.environ["DATAAGENT_MODEL"] = args.model

    model = os.environ.get("DATAAGENT_MODEL", "gemini-flash-latest")

    examples = _load_dev(limit=args.limit, db_id=args.db_id,
                         difficulty=args.difficulty)
    if not examples:
        print("No examples matched filter.", file=sys.stderr)
        sys.exit(2)

    print(f"▶ Spider eval: {len(examples)} examples, model={model}")
    results = run_spider_eval(examples)

    summary = summarize(results, model)
    report = render_report(results, summary)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=str)
    )
    (out_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2)
    )
    (out_dir / "report.md").write_text(report)

    error_results = [r for r in results if r.error]
    if error_results:
        (out_dir / "errors.json").write_text(
            json.dumps([asdict(r) for r in error_results], indent=2, default=str)
        )

    print()
    print(f"✓ Execution accuracy: {summary.execution_accuracy:.1%} "
          f"({summary.n_correct}/{summary.n_total})")
    print(f"✓ Errors: {summary.n_error}")
    print(f"✓ Avg latency: {summary.avg_latency_s:.2f}s")
    print(f"✓ Reports: {out_dir}/report.md")


if __name__ == "__main__":
    main()
