"""Write the compiled LangGraph as Mermaid so docs can't drift from code.

    uv run python scripts/export_graph.py          # writes docs/graph.mmd
    uv run python scripts/export_graph.py --check  # exit 1 if docs/graph.mmd is stale

The README embeds the same diagram between the GRAPH:BEGIN/END markers;
this script rewrites that block too.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.orchestrator import build_graph  # noqa: E402

MMD_PATH = ROOT / "docs" / "graph.mmd"
README_PATH = ROOT / "README.md"
_README_BLOCK = re.compile(r"(<!-- GRAPH:BEGIN -->\n)(.*?)(<!-- GRAPH:END -->)", re.DOTALL)


def render() -> str:
    # The interactive graph (API) is a superset of the plain one (app,
    # benchmarks): it adds the clarify and confirm_cost interrupt nodes.
    return build_graph(interactive=True).compile().get_graph().draw_mermaid()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if outputs are stale")
    args = parser.parse_args()

    mermaid = render()
    readme = README_PATH.read_text()
    new_readme = _README_BLOCK.sub(
        lambda m: f"{m.group(1)}```mermaid\n{mermaid.rstrip()}\n```\n{m.group(3)}", readme
    )

    if args.check:
        stale = (not MMD_PATH.exists() or MMD_PATH.read_text() != mermaid) or new_readme != readme
        if stale:
            print("docs/graph.mmd or README graph block is stale; run scripts/export_graph.py")
            return 1
        return 0

    MMD_PATH.write_text(mermaid)
    README_PATH.write_text(new_readme)
    print(f"wrote {MMD_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
