"""Router: features, fitting, and offline cost/accuracy replay."""

import numpy as np

from routing.router import FEATURES, Router, featurize, fit, simulate


def test_featurize_shape_and_cues():
    x = featurize("Which singers never performed in more than 2 concerts per year?", ["singer", "concert"], 12)
    assert len(x) == len(FEATURES)
    f = dict(zip(FEATURES, x))
    assert f["negation"] == 1 and f["compare"] == 1 and f["per_group"] == 1 and f["tables_mentioned"] == 2


def test_fit_learns_signal_and_simulation_trades_cost_for_accuracy(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    hard = rng.random(n) < 0.3
    X = np.column_stack([hard + rng.normal(0, 0.3, n), rng.normal(0, 1, (n, len(FEATURES) - 1))])
    cheap_ok = np.where(hard, rng.random(n) < 0.2, rng.random(n) < 0.9)
    strong_ok = np.where(hard, rng.random(n) < 0.7, rng.random(n) < 0.95)
    r = fit(X[:200], (~cheap_ok[:200]).astype(int), "cheap", "strong")
    curve = simulate(r, X[200:], cheap_ok[200:], strong_ok[200:], np.full(200, 0.001), np.full(200, 0.01),
                     thresholds=[0.0, 0.5, 1.0])
    all_strong, mid, all_cheap = curve
    assert all_strong["share_strong"] == 1.0 and all_cheap["share_strong"] == 0.0
    assert all_cheap["cost_mean"] < mid["cost_mean"] < all_strong["cost_mean"]
    assert mid["ex"] > all_cheap["ex"]  # routing the predicted-hard questions pays off
    path = tmp_path / "r.json"
    r.save(path)
    assert Router.load(path).pick(list(X[0])) in ("cheap", "strong")
