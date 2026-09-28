"""DataAgent command line.

    dataagent ask data.csv "top 5 regions by revenue"
    dataagent ask shop.sqlite "which customers never ordered?" --json
    dataagent profile data.parquet
    dataagent mcp                      # run the MCP server on stdio

Install from the repo: `uv tool install .` (or `uv run dataagent ...`).
CSV/Parquet files are loaded into DuckDB; .sqlite/.db files are opened
read-only. Needs GOOGLE_API_KEY (or DATAAGENT_MODEL=ollama/<model>).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _open(path: str):
    p = Path(path).expanduser()
    if not p.is_file():
        sys.exit(f"dataagent: no such file: {p}")
    if p.suffix.lower() in (".sqlite", ".db", ".sqlite3"):
        from core.database import Database
        return Database.from_sqlite(str(p), name=p.stem)
    if p.suffix.lower() in (".csv", ".parquet"):
        from core.duckdb_engine import DuckDBEngine
        eng = DuckDBEngine(name=p.stem)
        eng.load_file(str(p), "".join(c if c.isalnum() else "_" for c in p.stem).strip("_").lower() or "data")
        eng.lock()
        return eng
    sys.exit("dataagent: supported files are .csv, .parquet, .sqlite, .db")


def cmd_ask(args) -> int:
    from agents.orchestrator import run_analysis

    engine = _open(args.file)
    state = run_analysis(args.question, None, engine, None)
    df = state.get("result_df")
    if args.json:
        print(json.dumps({
            "sql": state.get("sql_query"), "error": state.get("error"),
            "columns": list(df.columns) if df is not None else [],
            "rows": df.head(args.rows).astype(object).where(df.head(args.rows).notna(), None).values.tolist()
            if df is not None else [],
            "narrative": state.get("narrative"),
            "warnings": (state.get("validation") or {}).get("warnings", []),
        }, default=str, indent=2))
        return 0 if not state.get("error") else 1
    print(f"SQL\n  {state.get('sql_query')}\n")
    if state.get("error"):
        print(f"Error: {state['error']}")
        return 1
    if df is not None:
        print(df.head(args.rows).to_string(index=False), "\n")
    if state.get("narrative"):
        print(state["narrative"])
    for w in (state.get("validation") or {}).get("warnings", []):
        print(f"! {w}")
    return 0


def cmd_profile(args) -> int:
    from core import injection, pii
    engine = _open(args.file)
    print(pii.mask_datasource(injection.sanitize_datasource(engine.datasource())).to_prompt())
    return 0


def cmd_mcp(_args) -> int:
    import mcp_server
    mcp_server.server.run("stdio")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dataagent", description="Ask questions about tabular data.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="answer a question about a file")
    a.add_argument("file")
    a.add_argument("question")
    a.add_argument("--json", action="store_true")
    a.add_argument("--rows", type=int, default=20)
    a.set_defaults(fn=cmd_ask)
    p = sub.add_parser("profile", help="show the schema as the model sees it")
    p.add_argument("file")
    p.set_defaults(fn=cmd_profile)
    m = sub.add_parser("mcp", help="run the MCP server on stdio")
    m.set_defaults(fn=cmd_mcp)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
