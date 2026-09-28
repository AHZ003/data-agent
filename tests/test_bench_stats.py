"""benchmarks/stats.py against hand-computed and textbook values."""

import math

import numpy as np
import pytest

from benchmarks.stats import bootstrap_ci, ci_gate, mcnemar_exact, paired_delta_ci, percentiles


def test_bootstrap_ci_matches_normal_approximation_at_spider_scale():
    n, k = 1034, 724  # 70.0% EX
    matches = [True] * k + [False] * (n - k)
    ci = bootstrap_ci(matches, n_boot=20_000)
    half = 1.96 * math.sqrt(ci.point * (1 - ci.point) / n)  # ~2.8 pts
    assert ci.point == pytest.approx(k / n)
    assert ci.lo == pytest.approx(ci.point - half, abs=0.004)
    assert ci.hi == pytest.approx(ci.point + half, abs=0.004)


def test_bootstrap_shortcut_equals_index_resampling():
    # The Binomial shortcut must agree with the naive resampling bootstrap.
    rng = np.random.default_rng(1)
    matches = rng.random(300) < 0.62
    naive = [rng.choice(matches, size=matches.size).mean() for _ in range(4000)]
    lo, hi = np.quantile(naive, [0.025, 0.975])
    ci = bootstrap_ci(matches, n_boot=20_000)
    assert ci.lo == pytest.approx(lo, abs=0.01)
    assert ci.hi == pytest.approx(hi, abs=0.01)


def test_bootstrap_is_deterministic_with_seed():
    m = [True, False, True, True]
    assert bootstrap_ci(m, seed=7) == bootstrap_ci(m, seed=7)


def test_bootstrap_degenerate_inputs():
    assert bootstrap_ci([True] * 10) == bootstrap_ci([True] * 10)
    ci = bootstrap_ci([True] * 10)
    assert ci.lo == ci.hi == 1.0
    assert math.isnan(bootstrap_ci([]).point)


def test_mcnemar_hand_computed():
    # 10 items only A gets right, 2 only B: p = 2 * P(X <= 2 | n=12, 0.5) = 158/4096
    a = [True] * 10 + [False] * 2 + [True] * 50 + [False] * 30
    b = [False] * 10 + [True] * 2 + [True] * 50 + [False] * 30
    r = mcnemar_exact(a, b)
    assert (r.a_only, r.b_only) == (10, 2)
    assert r.p_value == pytest.approx(158 / 4096)


def test_mcnemar_identical_configs_p_is_one():
    a = [True, False, True]
    assert mcnemar_exact(a, a).p_value == 1.0


def test_mcnemar_rejects_unpaired():
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])


def test_paired_delta_point_and_sign():
    a = [False] * 20 + [True] * 80
    b = [True] * 20 + [True] * 80  # B fixes 20 items, breaks none
    ci = paired_delta_ci(a, b)
    assert ci.point == pytest.approx(0.20)
    assert ci.lo > 0


def test_percentiles():
    p = percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5)
    assert p["p95"] == pytest.approx(95.05)


def test_ci_gate_ignores_noise_but_catches_real_drops():
    n = 200
    baseline = 0.70
    noisy = [True] * 136 + [False] * (n - 136)  # 68%: within noise
    assert ci_gate(noisy, baseline)[0] is True
    dropped = [True] * 110 + [False] * (n - 110)  # 55%: real drop
    assert ci_gate(dropped, baseline)[0] is False
