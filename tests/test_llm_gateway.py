"""LLM gateway: model routing, Ollama backend, per-agent models, cost accounting."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core import llm, tracing


@pytest.fixture
def fake_ollama(monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            out = json.dumps({"response": "SELECT 1", "prompt_eval_count": 120, "eval_count": 8}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{server.server_port}")
    yield seen
    server.shutdown()


def test_ollama_routing_and_usage(fake_ollama, monkeypatch):
    monkeypatch.setenv("OLLAMA_USD_PER_HOUR", "3600")  # $1 per second, to see it
    with tracing.usage_scope() as u:
        out = llm.generate("prompt", model="ollama/qwen2.5-coder:7b", max_output_tokens=64)
    assert out == "SELECT 1"
    assert fake_ollama[0]["model"] == "qwen2.5-coder:7b"
    assert fake_ollama[0]["options"]["num_predict"] == 64
    assert (u.tokens_in, u.tokens_out) == (120, 8)
    assert 0 < u.cost_usd < 5  # hardware time, not token pricing


def test_ollama_has_no_token_price():
    assert tracing._cost_usd("ollama/anything", 10**6, 10**6) == 0.0


def test_per_agent_model_override(monkeypatch):
    import config
    monkeypatch.setenv("DATAAGENT_MODEL_CODER", "ollama/sqlcoder")
    assert config.model_for("coder") == "ollama/sqlcoder"
    assert config.model_for("planner") == config.MODEL_NAME


def test_coder_uses_its_configured_model(fake_ollama, monkeypatch):
    from agents import coder_agent
    from core.datasource import DataSource
    import pandas as pd
    monkeypatch.setenv("DATAAGENT_MODEL_CODER", "ollama/local-coder")
    ds = DataSource.from_dataframes({"t": pd.DataFrame({"a": [1]})})
    assert coder_agent._generate_sql("q", ds) == "SELECT 1"
    assert fake_ollama[-1]["model"] == "local-coder"
