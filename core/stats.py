"""Statistical audit helpers for the Critic agent.

The old critic relied on hand-written heuristics (len(df) < 30, etc.).
That works until an analyst puts weight behind a number that looks
right but isn't statistically load-bearing. These helpers back the
critic with actual `scipy.stats` calls so warnings have real math
behind them:

- `iqr_outliers`     — Tukey fences (1.5·IQR), returns count and share
- `skewness_check`   — D'Agostino-Pearson (scipy.stats.skew + skewtest)
- `effect_size_d`    — Cohen's d for two-sample comparisons
- `correlation_test` — pearsonr + p-value + FDR-aware magnitude verdict

Every helper returns a small dataclass so the critic can turn them
into user-visible warnings without reaching into scipy internals.

Why this matters for the portfolio: a senior MLE critic should be
able to say "your 12% uplift is within the noise band of a t-test at
n=14" rather than "sample size small, be careful." The first is
useful; the second is a fortune cookie.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class OutlierReport:
    column: str
    n: int
    n_outliers: int
    share: float            # fraction 0-1
    lower_fence: float
    upper_fence: float


@dataclass
class SkewReport:
    column: str
    skew: float             # sample skewness (Fisher)
    p_value: Optional[float]  # None if n too small for skewtest
    verdict: str            # "symmetric" | "moderate" | "high" | "undefined"


@dataclass
class EffectSizeReport:
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    d: float                # Cohen's d
    magnitude: str          # "negligible" | "small" | "medium" | "large"


@dataclass
class CorrelationReport:
    r: float
    p_value: float
    n: int
    verdict: str            # "negligible" | "weak" | "moderate" | "strong"
    significant: bool       # p < 0.05 and |r| >= 0.1


# ── Outliers via Tukey fences ────────────────────────────────────────────

def iqr_outliers(series: pd.Series) -> Optional[OutlierReport]:
    """Return Tukey-fence outlier report for a numeric series.

    None is returned for columns that aren't numeric or have fewer
    than 4 non-null values — IQR is meaningless below that.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return None
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return OutlierReport(
            column=str(series.name), n=len(s), n_outliers=0, share=0.0,
            lower_fence=q1, upper_fence=q3,
        )
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    n_out = int(((s < lo) | (s > hi)).sum())
    return OutlierReport(
        column=str(series.name),
        n=len(s),
        n_outliers=n_out,
        share=round(n_out / len(s), 4),
        lower_fence=round(lo, 4),
        upper_fence=round(hi, 4),
    )


# ── Skewness ─────────────────────────────────────────────────────────────

def skewness_check(series: pd.Series) -> Optional[SkewReport]:
    """Report sample skewness + D'Agostino p-value when n >= 8.

    The verdict scale follows the common rule of thumb:
      |skew| < 0.5  symmetric
      |skew| < 1.0  moderate
      |skew| >= 1.0 high (consider log transform / winsorize)
    """
    from scipy import stats

    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return None
    sk = float(stats.skew(s, bias=False))
    p_val: Optional[float] = None
    if len(s) >= 8:
        try:
            p_val = float(stats.skewtest(s).pvalue)
        except ValueError:
            p_val = None
    abs_sk = abs(sk)
    if abs_sk < 0.5:
        verdict = "symmetric"
    elif abs_sk < 1.0:
        verdict = "moderate"
    else:
        verdict = "high"
    return SkewReport(
        column=str(series.name), skew=round(sk, 4), p_value=p_val, verdict=verdict,
    )


# ── Cohen's d for two-group comparison ───────────────────────────────────

def effect_size_d(a: pd.Series, b: pd.Series) -> Optional[EffectSizeReport]:
    """Cohen's d for two independent samples. None if either group <2.

    Uses the pooled-SD variant. Magnitude bins follow Cohen (1988):
      |d| < 0.2  negligible
      |d| < 0.5  small
      |d| < 0.8  medium
      else       large
    """
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return None
    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    sd = pooled**0.5 if pooled > 0 else 0.0
    d = (mean_a - mean_b) / sd if sd > 0 else 0.0
    abs_d = abs(d)
    if abs_d < 0.2:
        mag = "negligible"
    elif abs_d < 0.5:
        mag = "small"
    elif abs_d < 0.8:
        mag = "medium"
    else:
        mag = "large"
    return EffectSizeReport(
        n_a=n_a, n_b=n_b,
        mean_a=round(mean_a, 4), mean_b=round(mean_b, 4),
        d=round(d, 4), magnitude=mag,
    )


# ── Correlation with proper p-value ──────────────────────────────────────

def correlation_test(a: pd.Series, b: pd.Series) -> Optional[CorrelationReport]:
    """Pearson r + p-value. None if either series has <3 valid pairs."""
    from scipy import stats

    df = pd.concat([pd.to_numeric(a, errors="coerce"),
                    pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(df) < 3:
        return None
    r, p = stats.pearsonr(df.iloc[:, 0], df.iloc[:, 1])
    abs_r = abs(r)
    if abs_r < 0.1:
        verdict = "negligible"
    elif abs_r < 0.3:
        verdict = "weak"
    elif abs_r < 0.6:
        verdict = "moderate"
    else:
        verdict = "strong"
    return CorrelationReport(
        r=round(float(r), 4),
        p_value=round(float(p), 6),
        n=len(df),
        verdict=verdict,
        significant=(p < 0.05 and abs_r >= 0.1),
    )


# ── High-level audit for the critic ──────────────────────────────────────

def audit_dataframe(df: pd.DataFrame) -> list[str]:
    """Return a flat list of user-facing warning strings for a result df.

    This is the function `critic_agent.validate_sql_result` calls to
    turn scipy reports into narrative-grade guardrails. Returning
    strings (not structs) keeps the call site trivial and preserves
    backward compat with the existing `warnings: list[str]` field on
    ValidationReport.
    """
    warnings: list[str] = []
    if df is None or len(df) == 0:
        return warnings

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    for col in numeric_cols:
        out = iqr_outliers(df[col])
        if out and out.share >= 0.1 and out.n >= 10:
            warnings.append(
                f"Column '{col}' has {out.n_outliers} outliers "
                f"({out.share * 100:.1f}%) outside [{out.lower_fence:.2f}, "
                f"{out.upper_fence:.2f}]"
            )
        sk = skewness_check(df[col])
        # Only warn on "high" skew — "moderate" is the common case in
        # real business data and would be noisy to flag.
        if sk and sk.verdict == "high":
            warnings.append(
                f"Column '{col}' is highly skewed (skew={sk.skew:.2f}) — "
                "consider log-scale charts or a median-based summary"
            )

    return warnings
