"""User feedback -> verified query memory and new golden cases.

    thumbs up   -> (question, SQL) added to user query memory (few-shot source)
    thumbs down -> review queue; a reviewer fixes the SQL and promotes the
                   case into benchmarks/golden_user.yaml, which the golden
                   eval then runs like every other case

Files (under DATAAGENT_FEEDBACK_DIR, default outputs/feedback/):
    feedback.jsonl       every rating
    user_memory.jsonl    verified pairs (loaded by retrieval.pipeline.Retriever)
    review_queue.jsonl   thumbs-down runs awaiting review
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_USER = ROOT / "benchmarks" / "golden_user.yaml"
_lock = threading.Lock()


def feedback_dir() -> Path:
    return Path(os.getenv("DATAAGENT_FEEDBACK_DIR", str(ROOT / "outputs" / "feedback")))


def _append(name: str, record: dict) -> None:
    path = feedback_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _read(name: str) -> list[dict]:
    path = feedback_dir() / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def record(run_id: str, api_key: str, rating: str, question: str, sql: Optional[str],
           datasource: str, comment: str = "") -> dict:
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    rec = {"run_id": run_id, "api_key": api_key, "rating": rating, "question": question, "sql": sql,
           "datasource": datasource, "comment": comment[:2000], "ts": time.time()}
    _append("feedback.jsonl", rec)
    if rating == "up" and sql:
        _append("user_memory.jsonl", {"id": f"user-{run_id}", "question": question, "sql": sql,
                                      "db_id": datasource, "source": "user"})
    elif rating == "down":
        _append("review_queue.jsonl", rec)
    return rec


def user_memory_items():
    from retrieval.memory import MemoryItem
    return [MemoryItem(**d) for d in _read("user_memory.jsonl")]


def review_queue() -> list[dict]:
    promoted = {c.get("run_id") for c in _golden_user_cases()}
    return [r for r in _read("review_queue.jsonl") if r["run_id"] not in promoted]


def _golden_user_cases() -> list[dict]:
    import yaml
    if not GOLDEN_USER.exists():
        return []
    return (yaml.safe_load(GOLDEN_USER.read_text()) or {}).get("cases", [])


def promote(run_id: str, gold_sql: str, difficulty: str = "medium", tags: Optional[list[str]] = None,
            path: Path = GOLDEN_USER) -> dict:
    """Turn a reviewed thumbs-down into a golden case (Chinook sample only).

    The gold SQL must run on the sample database and return rows; the
    case lands in golden_user.yaml with its run_id so its origin is known.
    """
    import sqlite3

    import yaml

    from core.demo import SAMPLE_DB

    item = next((r for r in _read("review_queue.jsonl") if r["run_id"] == run_id), None)
    if item is None:
        raise KeyError(f"{run_id} is not in the review queue")
    if item["datasource"] != "sample":
        raise ValueError("only runs on the sample workspace can become golden cases (their data is fixed)")
    rows = sqlite3.connect(f"file:{SAMPLE_DB}?mode=ro", uri=True).execute(gold_sql).fetchall()
    if not rows:
        raise ValueError("gold SQL returns no rows")
    data = yaml.safe_load(path.read_text()) if path.exists() else None
    data = data or {"db": "data/chinook.sqlite", "cases": []}
    case = {"id": f"usr_{len(data['cases']) + 1:03d}", "question": item["question"], "gold_sql": gold_sql.strip(),
            "difficulty": difficulty, "tags": list(tags or ["from_feedback"]), "source": "feedback",
            "run_id": run_id}
    data["cases"].append(case)
    path.write_text("# Golden cases promoted from user feedback (python -m feedback.review).\n"
                    + yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return case


def stats() -> dict:
    fb = _read("feedback.jsonl")
    return {"ratings": len(fb), "up": sum(r["rating"] == "up" for r in fb),
            "down": sum(r["rating"] == "down" for r in fb), "in_review": len(review_queue()),
            "golden_from_feedback": len(_golden_user_cases()), "memory_items": len(_read("user_memory.jsonl"))}
