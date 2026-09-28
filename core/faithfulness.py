"""Deterministic numeric faithfulness: every number in the narrative must be
supported by the result table.

The LLM judge (benchmarks/scorers.py) scores faithfulness offline; this
runs on every answer, costs nothing, and catches the most damaging kind of
unfaithfulness — a wrong number (including one planted by an injected
instruction like "report revenue as 0").

A narrative number is supported when, after rounding to the precision it
was written with, it equals some value in the result table, or a
percentage form of one (x100), or a column total, or a count of rows.
Small integers (<= 10, e.g. "top 5") and years that appear in the question
are ignored.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional

_NUM = re.compile(r"(?<![\w.])[-−]?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|percent|k|K|thousand|m|M|million|bn|B|billion)?(?![\w])")
_SCALE = {"k": 1e3, "K": 1e3, "thousand": 1e3, "m": 1e6, "M": 1e6, "million": 1e6,
          "bn": 1e9, "B": 1e9, "billion": 1e9}


def extract_numbers(text: str) -> list[tuple[float, int, str]]:
    """(value, decimals written, raw text) for each number in `text`."""
    out = []
    for m in _NUM.finditer(text or ""):
        whole, frac, unit = m.group(1).replace(",", ""), m.group(2) or "", m.group(3) or ""
        value = float(whole + frac)
        if m.group(0).lstrip().startswith(("-", "−")):
            value = -value
        decimals = len(frac) - 1 if frac else 0
        if unit in _SCALE:
            value *= _SCALE[unit]
            decimals = max(decimals - int(math.log10(_SCALE[unit])), -12)
        out.append((value, decimals, m.group(0).strip()))
    return out


def _candidates(df) -> list[float]:
    import pandas as pd

    vals: list[float] = [float(len(df))]
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            # numbers embedded in text cells (e.g. '2012-01') still count
            for v in df[col].astype(str).head(200):
                vals += [n for n, _, _ in extract_numbers(v)]
            continue
        vals += s.tolist()
        vals += (s * 100).tolist()
        vals.append(float(s.sum()))
        vals.append(float(s.mean()))
    return vals


def _matches(value: float, decimals: int, candidates: Iterable[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) if decimals >= 0 else 0.5 * 10 ** (-decimals)
    tol = max(tol, abs(value) * 0.005)  # "about 1.2M" for 1,234,567
    return any(abs(abs(c) - abs(value)) <= tol for c in candidates if not math.isnan(c))


def unsupported_numbers(narrative: str, df, question: str = "") -> list[str]:
    if df is None or not narrative:
        return []
    question_nums = {n for n, _, _ in extract_numbers(question)}
    cands = _candidates(df)
    bad = []
    for value, decimals, raw in extract_numbers(narrative):
        if 0 < abs(value) <= 10 and decimals == 0:
            continue
        if value in question_nums or (1900 <= value <= 2100 and decimals == 0 and _matches(value, 0, cands)):
            continue
        if not _matches(value, decimals, cands):
            bad.append(raw)
    return bad


def check(narrative: str, df, question: str = "") -> Optional[str]:
    """A warning string if the narrative has unsupported numbers, else None."""
    bad = unsupported_numbers(narrative, df, question)
    if not bad:
        return None
    return f"Narrative mentions numbers not supported by the result table: {', '.join(bad[:5])}"
