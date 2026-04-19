"""Unit tests for core/stats.py — the scipy-backed statistical audit.

Covers the four primitives (iqr_outliers, skewness_check, effect_size_d,
correlation_test) and the top-level `audit_dataframe` that wraps them
into user-facing warning strings.
"""

import numpy as np
import pandas as pd
import pytest

from core.stats import (
    iqr_outliers,
    skewness_check,
    effect_size_d,
    correlation_test,
    audit_dataframe,
)


# ── iqr_outliers ─────────────────────────────────────────────────────────

def test_iqr_finds_obvious_outlier():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000], name="x")
    r = iqr_outliers(s)
    assert r is not None
    assert r.n_outliers == 1
    assert r.share > 0
    assert r.upper_fence < 1000


def test_iqr_clean_series_no_outliers():
    r = iqr_outliers(pd.Series(list(range(1, 21)), name="x"))
    assert r is not None and r.n_outliers == 0


def test_iqr_returns_none_on_short_series():
    assert iqr_outliers(pd.Series([1, 2], name="x")) is None


def test_iqr_handles_zero_variance():
    """A constant series has IQR=0; we should return a no-outlier report, not crash."""
    r = iqr_outliers(pd.Series([5, 5, 5, 5, 5], name="c"))
    assert r is not None and r.n_outliers == 0


# ── skewness_check ───────────────────────────────────────────────────────

def test_skew_symmetric_verdict():
    rng = np.random.default_rng(42)
    s = pd.Series(rng.normal(0, 1, 200), name="gauss")
    r = skewness_check(s)
    assert r is not None and r.verdict == "symmetric"


def test_skew_high_verdict_on_lognormal():
    rng = np.random.default_rng(42)
    s = pd.Series(np.exp(rng.normal(0, 1, 200)), name="income")
    r = skewness_check(s)
    assert r is not None and r.verdict == "high"
    # A sample of 200 from a log-normal distribution should be
    # statistically significantly skewed.
    assert r.p_value is not None and r.p_value < 0.01


def test_skew_too_few_values():
    assert skewness_check(pd.Series([1.0, 2.0], name="x")) is None


# ── effect_size_d ────────────────────────────────────────────────────────

def test_cohens_d_large_effect():
    a = pd.Series([100, 102, 98, 101, 99] * 4)
    b = pd.Series([80, 82, 78, 81, 79] * 4)
    r = effect_size_d(a, b)
    assert r is not None and r.magnitude == "large" and r.d > 0


def test_cohens_d_negligible_effect():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(100, 15, 50))
    b = pd.Series(rng.normal(100, 15, 50))
    r = effect_size_d(a, b)
    assert r is not None
    assert r.magnitude in {"negligible", "small"}


def test_cohens_d_too_small():
    r = effect_size_d(pd.Series([1.0]), pd.Series([2.0, 3.0]))
    assert r is None


# ── correlation_test ─────────────────────────────────────────────────────

def test_correlation_strong_positive():
    x = pd.Series(list(range(50)))
    y = x * 2 + 1
    r = correlation_test(x, y)
    assert r is not None
    assert r.verdict == "strong" and r.significant and r.r > 0.99


def test_correlation_negligible_random():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(0, 1, 60))
    y = pd.Series(rng.normal(0, 1, 60))
    r = correlation_test(x, y)
    assert r is not None
    # With n=60 uncorrelated noise, r should be small and NOT
    # declared significant by the combined p<0.05 AND |r|>=0.1 gate.
    assert r.verdict in {"negligible", "weak"}


def test_correlation_too_few_points():
    assert correlation_test(pd.Series([1, 2]), pd.Series([3, 4])) is None


# ── audit_dataframe end-to-end ───────────────────────────────────────────

def test_audit_flags_high_skew_column():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "normal_amount": rng.normal(100, 15, 100),
        "income": np.exp(rng.normal(0, 1, 100)),
    })
    warnings = audit_dataframe(df)
    assert any("income" in w and "skewed" in w for w in warnings)
    assert not any("normal_amount" in w and "skewed" in w for w in warnings)


def test_audit_flags_outlier_column_large_sample():
    # Need >= 10 rows and >= 10% outlier share for the warning to fire.
    vals = list(range(1, 81)) + [100_000] * 10  # 10/90 = 11% outliers
    df = pd.DataFrame({"amount": vals})
    warnings = audit_dataframe(df)
    assert any("amount" in w and "outliers" in w for w in warnings)


def test_audit_empty_dataframe_quiet():
    assert audit_dataframe(pd.DataFrame()) == []


def test_audit_handles_non_numeric_columns():
    df = pd.DataFrame({"category": ["a", "b", "c"], "n": [1, 2, 3]})
    # Just shouldn't crash on non-numeric columns.
    warnings = audit_dataframe(df)
    assert isinstance(warnings, list)
