"""CLI: profile and ask (LLM stubbed)."""

import json
from unittest.mock import patch

import pandas as pd

import cli
from agents import coder_agent, planner_agent, storyteller_agent
from models.analysis_plan import AnalysisPlan, AnalysisType, PlanStep


def test_profile_masks_pii(capsys):
    assert cli.main(["profile", "data/chinook.sqlite"]) == 0
    out = capsys.readouterr().out
    assert 'Table "Customer"' in out and "@" not in out.split('Table "Customer"')[1].split("Table")[0]


def test_ask_json_on_csv(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DATAAGENT_TRACE_PATH", str(tmp_path / "t.jsonl"))
    csv = tmp_path / "sales.csv"
    pd.DataFrame({"region": ["West", "East"], "amount": [5, 7]}).to_csv(csv, index=False)
    plan = AnalysisPlan(question="q", analysis_types=[AnalysisType.DESCRIPTIVE], steps=[PlanStep(agent="coder", task="t")])
    with patch.object(coder_agent, "_generate_sql", return_value='SELECT SUM("amount") AS total FROM "sales"'), \
         patch.object(planner_agent, "create_analysis_plan", return_value=plan), \
         patch.object(storyteller_agent, "generate_narrative", return_value="Total is 12."):
        assert cli.main(["ask", str(csv), "total amount?", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["rows"] == [[12]] and out["narrative"] == "Total is 12."
