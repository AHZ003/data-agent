"""MCP server tools, called through the MCP server interface with the LLM stubbed."""

import asyncio
import json
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("mcp")

import mcp_server  # noqa: E402
from agents import coder_agent, planner_agent, storyteller_agent  # noqa: E402
from models.analysis_plan import AnalysisPlan, AnalysisType, PlanStep  # noqa: E402


def _call(name, **args):
    res = asyncio.run(mcp_server.server.call_tool(name, args))
    if getattr(res, "isError", False) or getattr(res, "is_error", False):
        raise RuntimeError(res.content[0].text)
    structured = getattr(res, "structuredContent", None) or getattr(res, "structured_content", None)
    if structured is not None:
        return structured.get("result", structured)
    return json.loads(res.content[0].text)


def test_tools_are_registered():
    names = {t.name for t in asyncio.run(mcp_server.server.list_tools())}
    assert names == {"list_datasources", "open_file", "profile", "ask", "get_run"}


def test_open_file_profile_and_ask(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAAGENT_TRACE_PATH", str(tmp_path / "t.jsonl"))
    csv = tmp_path / "sales.csv"
    pd.DataFrame({"region": ["West", "East"], "amount": [5, 7], "email": ["a@x.io", "b@y.io"]}).to_csv(csv, index=False)
    opened = _call("open_file", path=str(csv))
    assert opened["tables"] == ["sales"]
    prof = _call("profile", datasource_id=opened["id"])
    assert "a@x.io" not in str(prof) and "<EMAIL>" in str(prof)

    plan = AnalysisPlan(question="q", analysis_types=[AnalysisType.DESCRIPTIVE], steps=[PlanStep(agent="coder", task="t")])
    with patch.object(coder_agent, "_generate_sql", return_value='SELECT SUM("amount") AS total FROM "sales"'), \
         patch.object(planner_agent, "create_analysis_plan", return_value=plan), \
         patch.object(storyteller_agent, "generate_narrative", return_value="Total is 12."):
        out = _call("ask", datasource_id=opened["id"], question="total amount?")
    assert out["rows"] == [[12]] and out["narrative"] == "Total is 12."
    assert _call("get_run", run_id=out["run_id"])["sql"].startswith("SELECT")
    assert any(d["id"] == "sample" for d in _call("list_datasources"))
