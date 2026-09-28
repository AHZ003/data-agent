"""Load test for the DataAgent API.

    # against a stub-LLM server (measures the service, not Gemini):
    uv run python loadtest/serve_stub.py --port 8010 &
    uvx locust -f loadtest/locustfile.py --host http://localhost:8010 \
        --headless -u 25 -r 5 -t 60s --csv outputs/loadtest/u25

    # against a real deployment (spends LLM quota):
    DATAAGENT_API_KEY=... uvx locust -f loadtest/locustfile.py --host https://<api-url> ...

Each simulated user asks a question from a realistic mix, reads the full
SSE stream, and sometimes asks a follow-up in the same conversation.
"""

import json
import os
import random

from locust import HttpUser, between, task

QUESTIONS = [
    "What is total revenue?",
    "Which 5 artists earned the most revenue?",
    "How did total revenue change year over year?",
    "Which genres sell best?",
    "How many customers are in each country?",
    "Which support rep's customers spend the most on average?",
    "How many tracks have never been purchased?",
    "What is the average invoice total by country?",
]
FOLLOW_UPS = ["now break it down by country", "only for 2012", "show the top 3"]


class Analyst(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        key = os.getenv("DATAAGENT_API_KEY")
        self.headers = {"X-API-Key": key} if key else {}

    def _ask(self, question, conversation_id=None, name="analyze"):
        body = {"datasource_id": "sample", "question": question}
        if conversation_id:
            body["conversation_id"] = conversation_id
        with self.client.post("/v1/analyze", json=body, headers=self.headers, name=name,
                              catch_response=True) as resp:
            if resp.status_code != 200 or "event: done" not in resp.text:
                resp.failure(f"status {resp.status_code}")
                return None
            done = resp.text.split("event: done\ndata: ", 1)[1].split("\n", 1)[0]
            return json.loads(done)["conversation_id"]

    @task(4)
    def ask(self):
        self._ask(random.choice(QUESTIONS))

    @task(1)
    def ask_with_follow_up(self):
        conv = self._ask(random.choice(QUESTIONS))
        if conv:
            self._ask(random.choice(FOLLOW_UPS), conv, name="analyze (follow-up)")
