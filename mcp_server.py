"""DataAgent as an MCP server: ask your data questions from Claude, IDEs, other agents.

    uv sync --extra mcp
    uv run python mcp_server.py            # stdio transport

Claude Desktop (claude_desktop_config.json):

    {"mcpServers": {"dataagent": {
        "command": "uv",
        "args": ["--directory", "/path/to/data-agent", "run", "python", "mcp_server.py"],
        "env": {"GOOGLE_API_KEY": "..."}}}}

Tools: list_datasources, open_file, profile, ask, get_run. Every query goes
through the same guarded, read-only pipeline as the app and the API; the
Chinook sample is preloaded as datasource "sample".
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from core import resources
from core.demo import SAMPLE_QUESTIONS, load_sample_workspace

server = MCPServer(
    name="dataagent",
    instructions=(
        "Answer analytics questions over tabular data with read-only SQL. Call list_datasources "
        "(or open_file for a local CSV/Parquet/SQLite file), then ask(datasource_id, question). "
        "Show the returned SQL to the user alongside the answer."
    ),
)

_names: dict[str, str] = {}
_runs: dict[str, dict] = {}


def _ensure_sample() -> None:
    if "sample" not in _names:
        resources.register(load_sample_workspace(), None, datasource_id="sample")
        _names["sample"] = "Chinook music store (sample)"


@server.tool(description="List queryable datasources with their tables.")
def list_datasources() -> list[dict]:
    _ensure_sample()
    out = []
    for ds_id, name in _names.items():
        engine, _ = resources.get(ds_id)
        out.append({"id": ds_id, "name": name, "tables": engine.datasource().table_names})
    return out


@server.tool(description="Open a local .csv/.parquet (loaded into DuckDB) or .sqlite/.db file (read-only). "
                         "Returns the datasource id.")
def open_file(path: str, name: str = "") -> dict:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    suffix = p.suffix.lower()
    if suffix in (".csv", ".parquet"):
        from core.duckdb_engine import DuckDBEngine
        engine = DuckDBEngine(name=name or p.stem)
        table = "".join(ch if ch.isalnum() else "_" for ch in p.stem).strip("_").lower() or "data"
        engine.load_file(str(p), table)
        engine.lock()
    elif suffix in (".sqlite", ".db", ".sqlite3"):
        from core.database import Database
        engine = Database.from_sqlite(str(p), name=name or p.stem)
    else:
        raise ValueError("supported: .csv, .parquet, .sqlite, .db")
    ds_id = resources.register(engine, None)
    _names[ds_id] = name or p.name
    return {"id": ds_id, "tables": engine.datasource().table_names}


@server.tool(description="Describe a datasource's tables, columns, keys and sample values "
                         "(PII masked), exactly as the SQL model sees it.")
def profile(datasource_id: str) -> str:
    from core import injection, pii
    _ensure_sample()
    engine, _ = resources.get(datasource_id)
    text = pii.mask_datasource(injection.sanitize_datasource(engine.datasource())).to_prompt()
    if datasource_id == "sample":
        text += "\n\nSuggested questions:\n" + "\n".join(f"- {q}" for q in SAMPLE_QUESTIONS)
    return text


@server.tool(description="Answer a question about a datasource. Returns the SQL that ran, the first 50 "
                         "result rows, a short narrative, and any validation warnings.")
def ask(datasource_id: str, question: str) -> dict[str, Any]:
    from agents.orchestrator import run_analysis
    _ensure_sample()
    engine, df = resources.get(datasource_id)
    state = run_analysis(question, None, engine, df)
    result = state.get("result_df")
    run_id = uuid.uuid4().hex[:12]
    out = {
        "run_id": run_id,
        "sql": state.get("sql_query"),
        "error": state.get("error"),
        "columns": list(result.columns) if result is not None else [],
        "rows": result.head(50).astype(object).where(result.head(50).notna(), None).values.tolist()
        if result is not None else [],
        "row_count": len(result) if result is not None else 0,
        "narrative": state.get("narrative"),
        "warnings": (state.get("validation") or {}).get("warnings", []),
    }
    _runs[run_id] = out
    return out


@server.tool(description="Fetch the result of an earlier ask() by run_id.")
def get_run(run_id: str) -> dict[str, Any]:
    if run_id not in _runs:
        raise ValueError(f"unknown run_id {run_id}")
    return _runs[run_id]


if __name__ == "__main__":
    server.run("stdio")
