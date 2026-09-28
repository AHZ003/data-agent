"""Online evaluation: score a sample of live runs, continuously.

Offline benchmarks say how the agent does on questions we wrote; this says
how it does on questions users actually ask. A DATAAGENT_ONLINE_EVAL_RATE
share of completed API runs (default 10%) is scored with the deterministic
checks — executed, non-empty result, critic not rejecting, narrative
numbers supported — plus, when DATAAGENT_ONLINE_JUDGE=1, the LLM
faithfulness judge. Records go to outputs/online_eval.jsonl, which the
alerts (monitoring/alerts.py) read.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
_lock = threading.Lock()


def path() -> Path:
    return Path(os.getenv("DATAAGENT_ONLINE_EVAL_PATH", str(ROOT / "outputs" / "online_eval.jsonl")))


def checks(result_df, error: Optional[str], validation: Optional[dict], faithfulness_warning: Optional[str]) -> dict:
    status = (validation or {}).get("status")
    c = {
        "executed": error is None and result_df is not None,
        "non_empty": result_df is not None and len(result_df) > 0,
        "critic_ok": status != "rejected",
        "numbers_supported": faithfulness_warning is None,
    }
    c["passed"] = all(c.values())
    return c


def maybe_score(run_id: str, question: str, sql: Optional[str], result_df, error, validation, narrative: str,
                faithfulness_warning: Optional[str], latency_s: float, cost_usd: float,
                rng: random.Random = random) -> Optional[dict]:
    rate = float(os.getenv("DATAAGENT_ONLINE_EVAL_RATE", "0.1"))
    if rng.random() >= rate:
        return None
    rec = {"ts": time.time(), "run_id": run_id, "question": question, "sql": sql,
           "latency_s": round(latency_s, 3), "cost_usd": cost_usd,
           **checks(result_df, error, validation, faithfulness_warning)}
    if os.getenv("DATAAGENT_ONLINE_JUDGE", "0") == "1" and narrative and result_df is not None:
        from benchmarks.scorers import score_narrative_faithfulness
        judged = score_narrative_faithfulness(question, result_df, narrative)
        rec["judge_score"] = judged.score
        rec["passed"] = rec["passed"] and judged.passed
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load(since: float = 0.0) -> list[dict]:
    p = path()
    if not p.exists():
        return []
    return [r for r in (json.loads(line) for line in p.read_text().splitlines() if line.strip()) if r["ts"] >= since]
