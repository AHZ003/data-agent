"""API end to end with the LLM stubbed: SSE, multi-turn, interrupts, edits, auth."""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agents import coder_agent, planner_agent, storyteller_agent
from models.analysis_plan import AnalysisPlan, AnalysisType, PlanStep


def _events(resp) -> list[tuple[str, dict]]:
    out = []
    for block in resp.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _plan(q, schema, clarify=None):
    return AnalysisPlan(question=q, analysis_types=[AnalysisType.DESCRIPTIVE],
                        steps=[PlanStep(agent="coder", task="sql")], clarifying_question=clarify)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAAGENT_CHECKPOINT_PATH", str(tmp_path / "ckpt.sqlite"))
    monkeypatch.setenv("DATAAGENT_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv("DATAAGENT_AUDIT_PATH", str(tmp_path / "audit.sqlite"))
    monkeypatch.delenv("DATAAGENT_API_KEYS", raising=False)
    from api import main
    monkeypatch.setattr(main, "EDITS_LOG", tmp_path / "edits.jsonl")
    main.S.__init__()
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def stub_llm():
    prompts = []

    def fake_sql(question, ds, error_context="", model=None, context=None):
        prompts.append(question)
        if "by country" in question:
            return 'SELECT "BillingCountry", SUM("Total") AS t FROM "Invoice" GROUP BY 1 ORDER BY t DESC'
        return 'SELECT SUM("Total") AS revenue FROM "Invoice"'

    with patch.object(coder_agent, "_generate_sql", side_effect=fake_sql), \
         patch.object(storyteller_agent, "stream_narrative", side_effect=lambda **k: iter(["Revenue ", "is up."])), \
         patch.object(planner_agent, "create_analysis_plan",
                      side_effect=lambda q, s: _plan(q, s, "Which metric defines best?" if "best" in q else None)):
        yield prompts


def test_analyze_streams_every_stage(client, stub_llm):
    r = client.post("/v1/analyze", json={"question": "What is total revenue?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _events(r)
    kinds = [e for e, _ in events]
    assert kinds[:3] == ["plan", "sql", "result_preview"]
    assert kinds.count("narrative_token") == 2 and kinds[-1] == "done"
    preview = dict(events)["result_preview"]
    assert preview["columns"] == ["revenue"] and preview["row_count"] == 1
    run = client.get(f"/v1/runs/{events[-1][1]['run_id']}").json()
    assert run["status"] == "done" and run["result"]["narrative"] == "Revenue is up."


def test_follow_up_sees_previous_turn(client, stub_llm):
    first = _events(client.post("/v1/analyze", json={"question": "What is total revenue?"}))
    conv = first[-1][1]["conversation_id"]
    client.post("/v1/analyze", json={"question": "now break it down by country", "conversation_id": conv})
    follow_up_prompt = stub_llm[-1]
    assert "CONVERSATION SO FAR" in follow_up_prompt and "What is total revenue?" in follow_up_prompt
    assert 'SUM("Total") AS revenue' in follow_up_prompt


def test_clarify_interrupt_and_resume(client, stub_llm):
    events = _events(client.post("/v1/analyze", json={"question": "Which one is the best?"}))
    assert events[-1][0] == "clarify"
    run_id = events[-1][1]["run_id"]
    assert client.get(f"/v1/runs/{run_id}").json()["status"] == "interrupted"
    resumed = _events(client.post(f"/v1/runs/{run_id}/resume", json={"answer": "by revenue"}))
    assert resumed[-1][0] == "done"
    assert "Clarification from the user: by revenue" in stub_llm[-1]
    assert client.post(f"/v1/runs/{run_id}/resume", json={"answer": "x"}).status_code == 409


def test_edit_sql_is_guarded_and_logged(client, stub_llm, tmp_path):
    run_id = _events(client.post("/v1/analyze", json={"question": "What is total revenue?"}))[-1][1]["run_id"]
    ok = client.post(f"/v1/runs/{run_id}/sql", json={"sql": 'SELECT COUNT(*) AS n FROM "Invoice"'})
    assert ok.status_code == 200 and ok.json()["result"]["rows"] == [[412]]
    bad = client.post(f"/v1/runs/{run_id}/sql", json={"sql": 'DELETE FROM "Invoice"'})
    assert bad.status_code == 400 and "SQLGuardError" in bad.json()["detail"]
    lines = (tmp_path / "edits.jsonl").read_text().splitlines()
    assert [json.loads(l)["ok"] for l in lines] == [True, False]


def test_upload_csv_creates_duckdb_workspace(client, stub_llm):
    csv = b"region,amount\nWest,5\nEast,7\n"
    r = client.post("/v1/datasources", files={"files": ("sales.csv", csv, "text/csv")})
    assert r.status_code == 201 and r.json()["tables"] == ["sales"]
    assert client.post("/v1/datasources", files={"files": ("x.txt", b"a", "text/plain")}).status_code == 400
    ids = [d["id"] for d in client.get("/v1/datasources").json()]
    assert "sample" in ids and r.json()["id"] in ids


def test_auth_and_rate_limit(client, stub_llm, monkeypatch):
    from api import main
    monkeypatch.setenv("DATAAGENT_API_KEYS", "alice:k-alice")
    assert client.get("/v1/datasources").status_code == 401
    assert client.get("/v1/datasources", headers={"X-API-Key": "k-alice"}).status_code == 200
    main.S.limiter = main.RateLimiter(per_minute=1, burst=1)
    h = {"X-API-Key": "k-alice"}
    assert client.post("/v1/analyze", json={"question": "q"}, headers=h).status_code == 200
    assert client.post("/v1/analyze", json={"question": "q"}, headers=h).status_code == 429


def test_token_bucket_refills():
    from api.auth import RateLimiter
    t = [0.0]
    rl = RateLimiter(per_minute=60, burst=2, clock=lambda: t[0])
    assert rl.allow("k") and rl.allow("k") and not rl.allow("k")
    t[0] += 1.0
    assert rl.allow("k") and not rl.allow("k")


def test_audit_records_runs_and_pii_columns(client, stub_llm):
    with patch.object(coder_agent, "_generate_sql",
                      return_value='SELECT c."Email", SUM(i."Total") FROM "Customer" c JOIN "Invoice" i '
                                   'ON c."CustomerId" = i."CustomerId" GROUP BY 1'):
        _events(client.post("/v1/analyze", json={"question": "spend per customer email"}))
    [row] = client.get("/v1/audit").json()
    assert row["question"] == "spend per customer email" and row["status"] == "ok"
    assert row["pii_columns_touched"] == ["customer.email"]
    assert row["engine"] == "Database"
