"""Drift alerts over online-eval records."""

import random

from monitoring.alerts import DAY, evaluate


def _recs(n, start, span, pass_rate, latency, cost, seed=0):
    rng = random.Random(seed)
    return [{"ts": start + rng.random() * span, "passed": rng.random() < pass_rate,
             "latency_s": latency * (0.8 + 0.4 * rng.random()), "cost_usd": cost} for _ in range(n)]


def test_quiet_when_stable():
    now = 10 * DAY
    recs = _recs(300, now - 8 * DAY, 7 * DAY, 0.8, 3.0, 0.001) + _recs(60, now - DAY, DAY, 0.78, 3.0, 0.001, 1)
    assert evaluate(recs, now=now)["alerts"] == []


def test_fires_on_real_drops_and_spikes():
    now = 10 * DAY
    recs = _recs(300, now - 8 * DAY, 7 * DAY, 0.8, 3.0, 0.001) + _recs(60, now - DAY, DAY, 0.4, 6.0, 0.003, 1)
    alerts = evaluate(recs, now=now)["alerts"]
    assert len(alerts) == 3
    assert any("pass rate" in a for a in alerts) and any("latency" in a for a in alerts)


def test_insufficient_data():
    assert evaluate(_recs(5, 0, DAY, 1, 1, 0), now=DAY)["status"] == "insufficient data"
