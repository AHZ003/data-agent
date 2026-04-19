"""Scoring functions for the DataAgent eval harness.

Each scorer takes the agent's output for a single case plus the case's
`expected` block and returns a `ScoreResult` with a 0-1 numeric score and
a short reason. The runner aggregates these into per-dimension and
overall scorecards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from config import GOOGLE_API_KEY, MODEL_NAME


@dataclass
class ScoreResult:
    name: str
    score: float  # 0.0 – 1.0
    passed: bool
    reason: str
    weight: float = 1.0


@dataclass
class CaseScorecard:
    case_id: str
    tags: list[str]
    duration_seconds: float
    error: Optional[str]
    scores: list[ScoreResult] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        if not self.scores:
            return 0.0
        total_w = sum(s.weight for s in self.scores) or 1.0
        return sum(s.score * s.weight for s in self.scores) / total_w

    @property
    def passed(self) -> bool:
        return self.error is None and all(s.passed for s in self.scores)


# ── Individual scorers ───────────────────────────────────────────────────

def score_sql_keywords(sql: Optional[str], must_contain: list[str]) -> ScoreResult:
    if not must_contain:
        return ScoreResult("sql_keywords", 1.0, True, "no requirement", weight=0.5)
    if not sql:
        return ScoreResult("sql_keywords", 0.0, False, "no SQL produced", weight=1.0)
    sql_l = sql.lower()
    hits = [kw for kw in must_contain if kw.lower() in sql_l]
    score = len(hits) / len(must_contain)
    return ScoreResult(
        "sql_keywords",
        score,
        passed=score >= 0.75,
        reason=f"matched {len(hits)}/{len(must_contain)}: {hits}",
        weight=1.0,
    )


def score_top_value(result_df: Optional[pd.DataFrame], expected: Optional[str]) -> ScoreResult:
    if expected is None:
        return ScoreResult("top_value", 1.0, True, "no requirement", weight=0.5)
    if result_df is None or len(result_df) == 0:
        return ScoreResult("top_value", 0.0, False, "empty result", weight=1.5)
    first_row = result_df.iloc[0]
    target = str(expected).strip().lower()
    cells = [str(v).strip().lower() for v in first_row.values]
    hit = any(target == c or target in c for c in cells)
    return ScoreResult(
        "top_value",
        1.0 if hit else 0.0,
        passed=hit,
        reason=f"top row = {first_row.to_dict()}, expected '{expected}'",
        weight=1.5,
    )


def score_row_count(
    result_df: Optional[pd.DataFrame],
    rmin: Optional[int],
    rmax: Optional[int],
) -> ScoreResult:
    if rmin is None and rmax is None:
        return ScoreResult("row_count", 1.0, True, "no bounds", weight=0.3)
    n = len(result_df) if result_df is not None else 0
    ok = True
    if rmin is not None and n < rmin:
        ok = False
    if rmax is not None and n > rmax:
        ok = False
    return ScoreResult(
        "row_count",
        1.0 if ok else 0.0,
        passed=ok,
        reason=f"got {n} rows (bounds {rmin}-{rmax})",
        weight=0.5,
    )


def score_chart_type(chart_config: Optional[dict], allowed: list[str]) -> ScoreResult:
    if not allowed:
        return ScoreResult("chart_type", 1.0, True, "no requirement", weight=0.3)
    actual = (chart_config or {}).get("chart_type", "")
    actual_str = getattr(actual, "value", str(actual))
    if "." in actual_str:  # strip enum class prefix like "ChartType.KPI_CARD"
        actual_str = actual_str.split(".")[-1]
    hit = actual_str.lower() in {a.lower() for a in allowed}
    return ScoreResult(
        "chart_type",
        1.0 if hit else 0.0,
        passed=hit,
        reason=f"chart_type={actual}, allowed={allowed}",
        weight=0.7,
    )


def score_narrative_keywords(narrative: Optional[str], must_mention: list[str]) -> ScoreResult:
    if not must_mention:
        return ScoreResult("narrative_keywords", 1.0, True, "no requirement", weight=0.3)
    if not narrative:
        return ScoreResult("narrative_keywords", 0.0, False, "no narrative", weight=1.0)
    nl = narrative.lower()
    hits = [k for k in must_mention if k.lower() in nl]
    score = len(hits) / len(must_mention)
    return ScoreResult(
        "narrative_keywords",
        score,
        passed=score >= 0.5,
        reason=f"matched {len(hits)}/{len(must_mention)}",
        weight=1.0,
    )


def score_confidence(validation: Optional[dict], min_conf: Optional[int]) -> ScoreResult:
    if min_conf is None:
        return ScoreResult("confidence", 1.0, True, "no requirement", weight=0.3)
    actual = (validation or {}).get("confidence", 0) or 0
    ok = actual >= min_conf
    return ScoreResult(
        "confidence",
        1.0 if ok else max(0.0, actual / max(min_conf, 1)),
        passed=ok,
        reason=f"critic confidence={actual} (min {min_conf})",
        weight=0.5,
    )


# ── Chaos / robustness scorers ───────────────────────────────────────────
# These are used by `chaos:` cases in cases.yaml where the "right" agent
# behavior is to *refuse gracefully*, warn loudly, or return empty — not
# to produce a confident answer. Expecting a top_value from a chaos case
# would reward hallucination; these scorers reward the opposite.

def score_expect_empty(result_df: Optional[pd.DataFrame], expect_empty: bool) -> ScoreResult:
    if not expect_empty:
        return ScoreResult("expect_empty", 1.0, True, "n/a", weight=0.0)
    n = len(result_df) if result_df is not None else 0
    ok = n == 0
    return ScoreResult(
        "expect_empty",
        1.0 if ok else 0.0,
        passed=ok,
        reason=f"got {n} rows; expected 0 (chaos / empty-filter case)",
        weight=1.5,
    )


def score_narrative_must_not_mention(narrative: Optional[str], banned: list[str]) -> ScoreResult:
    """Penalize hallucinated terms — e.g. inventing a column name the
    chaos question baits the agent into, or naming a 'top' value when
    the result set is empty."""
    if not banned:
        return ScoreResult("narrative_no_hallucinate", 1.0, True, "n/a", weight=0.0)
    if not narrative:
        return ScoreResult("narrative_no_hallucinate", 1.0, True, "no narrative", weight=1.0)
    nl = narrative.lower()
    hits = [b for b in banned if b.lower() in nl]
    return ScoreResult(
        "narrative_no_hallucinate",
        0.0 if hits else 1.0,
        passed=not hits,
        reason=f"hallucinated banned terms: {hits}" if hits else "no banned terms",
        weight=1.5,
    )


def score_expect_warning(validation: Optional[dict], expect: bool) -> ScoreResult:
    if not expect:
        return ScoreResult("expect_warning", 1.0, True, "n/a", weight=0.0)
    warnings = ((validation or {}).get("warnings") or [])
    ok = len(warnings) > 0
    return ScoreResult(
        "expect_warning",
        1.0 if ok else 0.0,
        passed=ok,
        reason=f"critic warnings = {len(warnings)}; expected ≥1 for chaos case",
        weight=1.0,
    )


# ── LLM-as-judge for narrative faithfulness ──────────────────────────────

_JUDGE_PROMPT = """You are an impartial judge scoring a data analyst's narrative.

QUESTION: {question}

NUMERIC RESULT (top rows of the answer table):
{result}

NARRATIVE TO JUDGE:
{narrative}

Score the narrative on FAITHFULNESS to the numeric result on a 0-100 scale:
- 100: every claim in the narrative is directly supported by the result table
- 70:  mostly supported, minor unsupported color
- 40:  mixed — some claims aren't in the data
- 0:   contradicts the data, or invents numbers not present

Return ONLY a JSON object: {{"score": <int 0-100>, "reason": "<one sentence>"}}
"""


def score_narrative_faithfulness(
    question: str,
    result_df: Optional[pd.DataFrame],
    narrative: Optional[str],
) -> ScoreResult:
    if not narrative or result_df is None or len(result_df) == 0:
        return ScoreResult(
            "faithfulness",
            0.0 if narrative else 1.0,
            passed=narrative is None,
            reason="no narrative or no result to judge against",
            weight=0.5,
        )
    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=GOOGLE_API_KEY)
        prompt = _JUDGE_PROMPT.format(
            question=question,
            result=result_df.head(10).to_string(),
            narrative=narrative[:1500],
        )
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=200),
        )
        text = resp.text.strip()
        if "{" in text:
            text = text[text.index("{") : text.rindex("}") + 1]
        data = json.loads(text)
        raw = int(data.get("score", 0))
        return ScoreResult(
            "faithfulness",
            raw / 100.0,
            passed=raw >= 60,
            reason=f"judge={raw} — {data.get('reason', '')[:120]}",
            weight=1.5,
        )
    except Exception as e:
        return ScoreResult(
            "faithfulness",
            0.5,
            passed=True,
            reason=f"judge unavailable ({type(e).__name__}); skipped",
            weight=0.0,
        )


# ── Top-level dispatcher ─────────────────────────────────────────────────

def score_case(
    case: dict,
    agent_output: dict[str, Any],
    duration: float,
    error: Optional[str],
    use_llm_judge: bool = True,
) -> CaseScorecard:
    """Apply all scorers and return a CaseScorecard."""
    expected = case.get("expected", {}) or {}
    sc = CaseScorecard(
        case_id=case["id"],
        tags=case.get("tags", []),
        duration_seconds=round(duration, 2),
        error=error,
    )

    is_chaos = "chaos" in (case.get("tags") or [])
    if error:
        # Chaos cases *allow* graceful errors (a quota-exhausted or
        # guard-blocked query isn't a crash in the product sense) —
        # we just don't reward them as hard as a clean run.
        weight = 1.0 if is_chaos else 2.0
        sc.scores.append(ScoreResult("execution", 0.0, False, error[:200], weight=weight))
        return sc

    sc.scores.append(ScoreResult("execution", 1.0, True, "ran without error", weight=2.0))
    sc.scores.append(score_sql_keywords(agent_output.get("sql_query"),
                                         expected.get("sql_must_contain", [])))
    sc.scores.append(score_top_value(agent_output.get("result_df"),
                                      expected.get("top_value")))
    sc.scores.append(score_row_count(agent_output.get("result_df"),
                                      expected.get("row_count_min"),
                                      expected.get("row_count_max")))
    sc.scores.append(score_chart_type(agent_output.get("chart_config"),
                                       expected.get("chart_type_in", [])))
    sc.scores.append(score_narrative_keywords(agent_output.get("narrative"),
                                               expected.get("narrative_must_mention", [])))
    sc.scores.append(score_confidence(agent_output.get("validation"),
                                       expected.get("min_confidence")))
    # Chaos / robustness assertions — only scored when the case opts in.
    if expected.get("expect_empty_result"):
        sc.scores.append(score_expect_empty(
            agent_output.get("result_df"), True))
    if expected.get("narrative_must_not_mention"):
        sc.scores.append(score_narrative_must_not_mention(
            agent_output.get("narrative"),
            expected.get("narrative_must_not_mention", []),
        ))
    if expected.get("expect_warnings_nonempty"):
        sc.scores.append(score_expect_warning(
            agent_output.get("validation"), True))
    if use_llm_judge:
        sc.scores.append(score_narrative_faithfulness(
            case["question"],
            agent_output.get("result_df"),
            agent_output.get("narrative"),
        ))
    return sc
