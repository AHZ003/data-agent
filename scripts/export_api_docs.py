"""Write docs/api.md from the FastAPI app's OpenAPI schema.

    uv run python scripts/export_api_docs.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "api.md"


def render() -> str:
    from api.main import app

    spec = app.openapi()
    lines = ["# HTTP API", "",
             "_Generated from the FastAPI app by `scripts/export_api_docs.py`; interactive docs at `/docs` "
             "on a running server._", "",
             "Auth: `X-API-Key` header (keys from `DATAAGENT_API_KEYS`; off when unset). Optional "
             "`X-Gemini-Key` runs the request on the caller's own Gemini key.", "",
             "| Method | Path | Summary |", "|---|---|---|"]
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            summary = " ".join((op.get("description") or op.get("summary") or "").split())
            lines.append(f"| `{method.upper()}` | `{path}` | {summary} |")
    lines += ["", "## Server-Sent Events from `/v1/analyze` and `/resume`", "",
              "| Event | Payload |", "|---|---|",
              "| `plan` | analysis types, steps, `clarifying_question` |",
              "| `sql` | `{sql, cache_hit}` |",
              "| `result_preview` | `{columns, rows (≤50), row_count}` |",
              "| `chart_spec` | Plotly figure JSON |",
              "| `validation` | critic status, confidence, warnings |",
              "| `narrative_token` | `{text}`, streamed |",
              "| `faithfulness` | `{warning}` when narrative numbers are not in the result |",
              "| `clarify` / `confirm_cost` | the run paused; answer with `POST /v1/runs/{run_id}/resume` |",
              "| `done` | `{run_id, conversation_id, error}` |",
              "| `error` | `{run_id, message}` |"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        return 0 if OUT.exists() and OUT.read_text() == text else 1
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
