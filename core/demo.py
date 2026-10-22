"""Public-demo guardrails and the preloaded sample workspace.

A stranger with the demo link must be able to try the app without
uploading anything, and must not be able to run up the bill:

- DEMO_MAX_QUESTIONS_PER_SESSION (default 10) caps questions on the
  server key per visitor per day (VisitorBudget; a reload starts a new
  session, so a per-session count alone is trivially reset). A visitor
  who pastes their own Gemini key is not capped.
- DEMO_MAX_UPLOAD_MB (default 10) caps uploads.
- The sample workspace is Chinook (data/chinook.sqlite, 11 tables),
  opened read-only, with suggested questions that exercise joins.

Unset DEMO_MODE (or set it to 0) to disable the limits locally.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.database import Database

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DB = ROOT / "data" / "chinook.sqlite"

SAMPLE_QUESTIONS = [
    "Which 5 artists earned the most revenue?",
    "How did total revenue change year over year?",
    "Which genres sell best in each country?",
    "Which support rep's customers spend the most on average?",
    "How many tracks have never been purchased?",
]


@dataclass(frozen=True)
class DemoLimits:
    enabled: bool
    max_questions: int
    max_upload_mb: int

    @classmethod
    def from_env(cls) -> "DemoLimits":
        return cls(
            enabled=os.getenv("DEMO_MODE", "0") not in ("0", "", "false", "False"),
            max_questions=int(os.getenv("DEMO_MAX_QUESTIONS_PER_SESSION", "10")),
            max_upload_mb=int(os.getenv("DEMO_MAX_UPLOAD_MB", "10")),
        )

    def question_allowed(self, asked_so_far: int, user_key: Optional[str] = None) -> tuple[bool, str]:
        """(allowed, message). Visitors using their own key are never capped."""
        if not self.enabled or (user_key and user_key.strip()):
            return True, ""
        if asked_so_far >= self.max_questions:
            return False, (
                f"The public demo allows {self.max_questions} questions per visitor per day on the shared key. "
                "Paste your own Gemini API key (free at aistudio.google.com/apikey) to keep going."
            )
        return True, ""

    def upload_allowed(self, size_bytes: int) -> tuple[bool, str]:
        if not self.enabled or size_bytes <= self.max_upload_mb * 1024 * 1024:
            return True, ""
        return False, f"The public demo accepts uploads up to {self.max_upload_mb} MB."


class VisitorBudget:
    """Questions used per visitor over a rolling window, in process memory.

    The visitor key is a hash of client IP + user agent. Kept in memory on
    purpose: the demo runs as a single Cloud Run instance, and a restart
    resetting the counts is acceptable for a spend guard.
    """

    def __init__(self, window_s: float = 86_400):
        self.window_s = window_s
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}

    @staticmethod
    def visitor_key(ip: str, user_agent: str = "") -> str:
        return hashlib.sha256(f"{ip}|{user_agent}".encode()).hexdigest()[:16]

    def _live(self, key: str, now: float) -> list[float]:
        hits = [t for t in self._hits.get(key, []) if now - t < self.window_s]
        self._hits[key] = hits
        return hits

    def used(self, key: str) -> int:
        with self._lock:
            return len(self._live(key, time.time()))

    def charge(self, key: str) -> int:
        with self._lock:
            now = time.time()
            hits = self._live(key, now)
            hits.append(now)
            return len(hits)


def load_sample_workspace() -> Database:
    """The Chinook sample database, read-only, ready for questions."""
    return Database.from_sqlite(str(SAMPLE_DB), name="chinook")
