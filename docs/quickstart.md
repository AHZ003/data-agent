# Quickstart

```bash
git clone https://github.com/AHZ003/data-agent.git && cd data-agent
make install                     # uv sync from uv.lock
cp .env.example .env             # set GOOGLE_API_KEY
```

| Interface | Command |
|---|---|
| Streamlit app | `make run` → http://localhost:8501 |
| HTTP API (SSE) | `make api` → http://localhost:8000/docs |
| CLI | `uv run dataagent ask data/chinook.sqlite "top 5 artists by revenue"` |
| MCP server | `uv sync --extra mcp && uv run python mcp_server.py` |
| Tests (no API key) | `make test` |
| Harness self-check (no API key) | `make bench-oracle` |
| Red team, SQL layer (no API key) | `make redteam-offline` |

Local models: `DATAAGENT_MODEL=ollama/qwen2.5-coder:7b` (any agent:
`DATAAGENT_MODEL_CODER=...`).
