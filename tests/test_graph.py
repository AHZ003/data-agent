"""Graph wiring: schema is a real node, and the docs diagram matches code."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from agents import orchestrator
from agents.schema_agent import profile_dataframe
from core.database import Database

ROOT = Path(__file__).resolve().parent.parent


def _state(schema=None):
    db = Database()
    df = pd.DataFrame({"region": ["a", "b"], "sales": [1.0, 2.0]})
    db.load_dataframe(df, "sales")
    return orchestrator._initial_state("q", schema, db, df, None)


def test_schema_is_the_entry_node():
    graph = orchestrator.build_graph().compile().get_graph()
    first = [e.target for e in graph.edges if e.source == "__start__"]
    assert first == ["schema"]
    assert any(e.source == "schema" and e.target == "planner" for e in graph.edges)


def test_schema_node_profiles_when_no_schema_given():
    state = orchestrator.schema_node(_state())
    assert state["schema"]["table_name"] == "sales"
    assert {c["name"] for c in state["schema"]["columns"]} == {"region", "sales"}
    assert state["schema"]["suggested_questions"] == []  # no LLM call in the graph
    assert state["agent_log"][-1]["agent_name"] == "Schema"


def test_schema_node_reuses_given_schema():
    df = pd.DataFrame({"x": [1, 2]})
    given = profile_dataframe(df, "given", suggest_questions=False)
    state = orchestrator.schema_node(_state(given))
    assert state["schema"]["table_name"] == "given"
    assert "Reused" in state["agent_log"][-1]["output_summary"]


def test_docs_graph_matches_compiled_graph():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_graph.py"), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
