"""Public-demo guardrails and the preloaded sample workspace.

A stranger with the demo link must be able to try the app without
uploading anything, and must not be able to run up the bill:

- DEMO_MAX_QUESTIONS_PER_SESSION (default 10) caps questions on the
  server key; a visitor who pastes their own Gemini key is not capped.
- DEMO_MAX_UPLOAD_MB (default 10) caps uploads.
- The sample workspace is Chinook (data/chinook.sqlite, 11 tables),
  opened read-only, with suggested questions that exercise joins.

Unset DEMO_MODE (or set it to 0) to disable the limits locally.
"""

from __future__ import annotations

import os
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
                f"The public demo allows {self.max_questions} questions per session on the shared key. "
                "Paste your own Gemini API key (free at aistudio.google.com/apikey) to keep going."
            )
        return True, ""

    def upload_allowed(self, size_bytes: int) -> tuple[bool, str]:
        if not self.enabled or size_bytes <= self.max_upload_mb * 1024 * 1024:
            return True, ""
        return False, f"The public demo accepts uploads up to {self.max_upload_mb} MB."


def load_sample_workspace() -> Database:
    """The Chinook sample database, read-only, ready for questions."""
    return Database.from_sqlite(str(SAMPLE_DB), name="chinook")
