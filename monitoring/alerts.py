"""Drift alerts: compare the last day of live runs with the week before.

    python -m monitoring.alerts            # exit 1 if any alert fires
    GET /v1/alerts                          # same, from the API

Three signals from outputs/online_eval.jsonl (core/online_eval.py):

  - pass rate: CI-aware, like the CI gate — alert only when the current
    window's upper 95% bound is below the baseline's point estimate;
  - p95 latency: alert above LATENCY_RATIO x baseline p95 (default 1.5);
  - cost per question: alert above COST_RATIO x baseline mean (default 1.5).

Needs MIN_RUNS scored runs in each window, else it reports "insufficient
data" instead of guessing. Schedule it (Cloud Scheduler -> /v1/alerts, or a
cron running this module) and page on a non-empty `alerts` list.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import numpy as np

DAY = 86400.0


def evaluate(records: list[dict], now: Optional[float] = None, window_s: float = DAY,
             baseline_s: float = 7 * DAY) -> dict:
    from benchmarks.stats import ci_gate

    now = now or time.time()
    min_runs = int(os.getenv("ALERT_MIN_RUNS", "20"))
    lat_ratio = float(os.getenv("ALERT_LATENCY_RATIO", "1.5"))
    cost_ratio = float(os.getenv("ALERT_COST_RATIO", "1.5"))
    cur = [r for r in records if now - window_s <= r["ts"] <= now]
    base = [r for r in records if now - window_s - baseline_s <= r["ts"] < now - window_s]
    out = {"window_runs": len(cur), "baseline_runs": len(base), "alerts": [], "status": "ok"}
    if len(cur) < min_runs or len(base) < min_runs:
        out["status"] = "insufficient data"
        return out

    base_pass = float(np.mean([r["passed"] for r in base]))
    ok, ci = ci_gate([r["passed"] for r in cur], base_pass)
    out["pass_rate"] = {"current": ci.point, "ci": [ci.lo, ci.hi], "baseline": base_pass}
    if not ok:
        out["alerts"].append(f"pass rate dropped: {ci.fmt()} vs baseline {base_pass:.1%}")

    cur_p95 = float(np.percentile([r["latency_s"] for r in cur], 95))
    base_p95 = float(np.percentile([r["latency_s"] for r in base], 95))
    out["p95_latency_s"] = {"current": cur_p95, "baseline": base_p95}
    if cur_p95 > lat_ratio * base_p95:
        out["alerts"].append(f"p95 latency {cur_p95:.1f}s vs baseline {base_p95:.1f}s")

    cur_cost = float(np.mean([r["cost_usd"] for r in cur]))
    base_cost = float(np.mean([r["cost_usd"] for r in base]))
    out["cost_per_question"] = {"current": cur_cost, "baseline": base_cost}
    if base_cost > 0 and cur_cost > cost_ratio * base_cost:
        out["alerts"].append(f"cost per question ${cur_cost:.5f} vs baseline ${base_cost:.5f}")

    out["status"] = "alert" if out["alerts"] else "ok"
    return out


def main() -> int:
    from core import online_eval
    report = evaluate(online_eval.load(since=time.time() - 8 * DAY))
    print(json.dumps(report, indent=2))
    return 1 if report["alerts"] else 0


if __name__ == "__main__":
    sys.exit(main())
