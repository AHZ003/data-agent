"""Statistics for benchmark results: CIs, paired tests, percentiles.

Why this module exists
----------------------
On Spider dev (n=1,034) a 95% CI on execution accuracy is roughly
±2.5 points. Many "improvements" to text-to-SQL systems are smaller
than that. Every accuracy number we report carries a bootstrap CI, and
every A/B between two configs on the same questions carries McNemar's
exact test.

Bootstrap shortcut
------------------
Resampling n Bernoulli outcomes with replacement and taking the mean is
distributed exactly as Binomial(n, p_hat) / n, so we draw from that
directly instead of materializing an (n_boot x n) index matrix. For a
paired difference d_i in {-1, 0, +1}, resampling is exactly a
multinomial over the three counts. Same distribution, O(n_boot) work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import binomtest


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float

    def fmt(self, pct: bool = True) -> str:
        if pct:
            return f"{self.point:.1%} [{self.lo:.1%}, {self.hi:.1%}]"
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


@dataclass(frozen=True)
class McNemar:
    a_only: int   # A correct, B wrong
    b_only: int   # B correct, A wrong
    p_value: float


def bootstrap_ci(
    matches: Sequence[bool], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> CI:
    """Percentile bootstrap CI for the mean of 0/1 outcomes."""
    n = len(matches)
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"))
    p = float(np.mean(np.asarray(matches, dtype=float)))
    rng = np.random.default_rng(seed)
    means = rng.binomial(n, p, size=n_boot) / n
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(p, float(lo), float(hi))


def paired_delta_ci(
    a: Sequence[bool], b: Sequence[bool], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> CI:
    """Bootstrap CI for mean(b) - mean(a) on the same items (paired)."""
    a_arr, b_arr = _paired(a, b)
    n = len(a_arr)
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"))
    d = b_arr.astype(int) - a_arr.astype(int)
    counts = np.array([(d == -1).sum(), (d == 0).sum(), (d == 1).sum()])
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, counts / n, size=n_boot)
    deltas = (draws[:, 2] - draws[:, 0]) / n
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return CI(float(d.mean()), float(lo), float(hi))


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> McNemar:
    """Exact (binomial) McNemar test for two configs on the same items.

    Only discordant pairs carry information: under H0 each discordant
    item is equally likely to favor A or B, so a_only ~ Binomial(a_only +
    b_only, 0.5). Two-sided p-value.
    """
    a_arr, b_arr = _paired(a, b)
    a_only = int((a_arr & ~b_arr).sum())
    b_only = int((~a_arr & b_arr).sum())
    n_disc = a_only + b_only
    p = 1.0 if n_disc == 0 else float(binomtest(a_only, n_disc, 0.5).pvalue)
    return McNemar(a_only=a_only, b_only=b_only, p_value=p)


def percentiles(values: Sequence[float], qs: Sequence[float] = (50, 95)) -> dict[str, float]:
    """{'p50': ..., 'p95': ...}; NaN for empty input."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {f"p{int(q)}": float("nan") for q in qs}
    return {f"p{int(q)}": float(np.percentile(arr, q)) for q in qs}


def ci_gate(current: Sequence[bool], baseline_point: float, **kw) -> tuple[bool, CI]:
    """CI-aware regression gate.

    Fails only when the current run's *upper* CI bound is below the
    baseline's point estimate — i.e. the drop is larger than sampling
    noise can explain. A plain `score < baseline` gate fails about half
    of all no-op PRs on a noisy eval; this one fails only on real drops.
    Returns (passed, current_ci).
    """
    ci = bootstrap_ci(current, **kw)
    return ci.hi >= baseline_point, ci


def _paired(a: Sequence[bool], b: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    if len(a) != len(b):
        raise ValueError(f"paired inputs differ in length: {len(a)} vs {len(b)}")
    return np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
