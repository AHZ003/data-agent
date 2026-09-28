"""Serve the real API with the LLM calls replaced by fixed simulated latency.

Measures what the service itself costs under concurrency (graph execution,
guarded SQL, checkpointer writes, SSE) without spending LLM quota. Every
LLM call site sleeps --llm-ms and returns canned output, so the numbers
are service overhead + simulated model time, not real Gemini latency.

    uv run python loadtest/serve_stub.py --port 8010 --llm-ms 800
"""

import argparse
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402

from agents import coder_agent, planner_agent, storyteller_agent  # noqa: E402
from models.analysis_plan import AnalysisPlan, AnalysisType, PlanStep  # noqa: E402

SQL = [
    'SELECT "BillingCountry", SUM("Total") AS revenue FROM "Invoice" GROUP BY 1 ORDER BY revenue DESC',
    'SELECT g."Name", SUM(il."UnitPrice" * il."Quantity") AS revenue FROM "InvoiceLine" il '
    'JOIN "Track" t ON il."TrackId" = t."TrackId" JOIN "Genre" g ON t."GenreId" = g."GenreId" GROUP BY 1',
    'SELECT strftime(\'%Y\', "InvoiceDate") AS year, SUM("Total") FROM "Invoice" GROUP BY 1 ORDER BY 1',
]


def main():
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)  # kill -USR1 <pid> dumps stacks

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--llm-ms", type=int, default=800, help="simulated latency per LLM call")
    args = parser.parse_args()
    delay = args.llm_ms / 1000

    def fake_sql(question, *a, **k):
        time.sleep(delay)
        return SQL[hash(question) % len(SQL)]

    def fake_plan(q, schema):
        time.sleep(delay)
        return AnalysisPlan(question=q, analysis_types=[AnalysisType.DESCRIPTIVE],
                            steps=[PlanStep(agent="coder", task="sql")])

    def fake_story(**kwargs):
        for word in "Revenue is concentrated in a few markets.".split():
            time.sleep(delay / 8)
            yield word + " "

    with patch.object(coder_agent, "_generate_sql", side_effect=fake_sql), \
         patch.object(planner_agent, "create_analysis_plan", side_effect=fake_plan), \
         patch.object(storyteller_agent, "stream_narrative", side_effect=fake_story):
        from api.main import app
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
