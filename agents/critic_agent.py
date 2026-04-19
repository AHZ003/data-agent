"""Critic Agent - Self-correction and validation of analysis outputs."""

import json
import re
from google import genai
from google.genai import types as genai_types
import pandas as pd
from typing import Optional, List

from config import (
    GOOGLE_API_KEY,
    MODEL_NAME,
    CRITIC_AGENT_SYSTEM_PROMPT,
    CRITIC_CONFIDENCE_THRESHOLD_WARN,
    CRITIC_CONFIDENCE_THRESHOLD_REJECT,
)
from models.analysis_plan import ValidationReport, ValidationStatus, SemanticSchema
from models.chart_config import ChartConfig, ChartType
from core import stats as da_stats


def validate_sql_result(
    df: Optional[pd.DataFrame],
    sql_query: str,
    question: str,
    schema: SemanticSchema,
) -> ValidationReport:
    """Validate SQL query results for correctness and reasonableness."""
    warnings: List[str] = []
    corrections: List[str] = []
    confidence = 90  # Start high, deduct for issues

    # Check empty result
    if df is None or len(df) == 0:
        return ValidationReport(
            status=ValidationStatus.REJECTED,
            confidence=20,
            warnings=["Query returned no results"],
            rejection_reason="Empty result set. The query may have incorrect filters or column names.",
        )

    # Check for unreasonable values
    for col in df.select_dtypes(include="number").columns:
        # Check for all-null numeric columns
        if df[col].isnull().all():
            warnings.append(f"Column '{col}' is entirely null")
            confidence -= 15

        # Check for suspiciously large values that might indicate sum of IDs
        if df[col].max() > 1e12:
            warnings.append(f"Column '{col}' has very large values ({df[col].max():.0f}) - possible aggregation of ID column")
            confidence -= 10

    # Check if result makes sense relative to question
    if len(df) == 1 and len(df.columns) == 1:
        # Single value result - usually OK for aggregation questions
        pass
    elif len(df) > 500:
        warnings.append(f"Result has {len(df)} rows - consider adding more specific filters")
        confidence -= 5

    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        warnings.append(f"Result contains {dup_count} duplicate rows")
        confidence -= 5

    # Scipy-backed statistical audit: IQR outliers + high-skew flags.
    # See core/stats.py — this replaces the old "len(df) < 30, be
    # careful" hand-waving with real math the critic can stand behind.
    try:
        stats_warnings = da_stats.audit_dataframe(df)
        for w in stats_warnings:
            warnings.append(w)
            confidence -= 5
    except Exception:
        # Stats audit is best-effort — never fail the agent run on it.
        pass

    # Determine status
    if confidence >= CRITIC_CONFIDENCE_THRESHOLD_WARN:
        status = ValidationStatus.VALIDATED
    elif confidence >= CRITIC_CONFIDENCE_THRESHOLD_REJECT:
        status = ValidationStatus.VALIDATED_WITH_WARNINGS
    else:
        status = ValidationStatus.REJECTED

    return ValidationReport(
        status=status,
        confidence=max(0, confidence),
        warnings=warnings,
        corrections_made=corrections,
        rejection_reason="Multiple data quality issues detected" if status == ValidationStatus.REJECTED else None,
    )


def validate_statistical_claim(
    sample_size: int,
    correlation: Optional[float] = None,
    pct_change: Optional[float] = None,
    base_value: Optional[float] = None,
    confidence_interval_width: Optional[float] = None,
    forecast_value: Optional[float] = None,
) -> ValidationReport:
    """Validate statistical claims for rigor."""
    warnings: List[str] = []
    confidence = 90

    # Sample size check
    if sample_size < 30:
        warnings.append(
            f"Sample size is only {sample_size}. Results may not be statistically significant (n < 30)."
        )
        confidence -= 20

    if sample_size < 10:
        confidence -= 20

    # Correlation strength
    if correlation is not None:
        abs_r = abs(correlation)
        if abs_r < 0.1:
            warnings.append(f"Correlation of {correlation:.3f} is negligible")
            confidence -= 15
        elif abs_r < 0.3:
            warnings.append(f"Correlation of {correlation:.3f} is weak")
            confidence -= 10

    # Percentage change from small base
    if pct_change is not None and base_value is not None:
        if base_value < 10 and abs(pct_change) > 100:
            warnings.append(
                f"A {pct_change:.0f}% change from a base of {base_value:.1f} may be misleading"
            )
            confidence -= 15

    # Prediction confidence interval
    if confidence_interval_width is not None and forecast_value is not None:
        if confidence_interval_width > abs(forecast_value) * 2:
            warnings.append(
                "Confidence interval is wider than the forecast value - prediction is very uncertain"
            )
            confidence -= 25

    if confidence >= CRITIC_CONFIDENCE_THRESHOLD_WARN:
        status = ValidationStatus.VALIDATED
    elif confidence >= CRITIC_CONFIDENCE_THRESHOLD_REJECT:
        status = ValidationStatus.VALIDATED_WITH_WARNINGS
    else:
        status = ValidationStatus.REJECTED

    return ValidationReport(
        status=status,
        confidence=max(0, confidence),
        warnings=warnings,
    )


def validate_chart(
    df: pd.DataFrame,
    chart_config: ChartConfig,
) -> ValidationReport:
    """Validate that the chart type is appropriate for the data."""
    warnings: List[str] = []
    corrections: List[str] = []
    confidence = 95

    # Pie chart with too many slices
    if chart_config.chart_type == ChartType.PIE:
        if chart_config.x_column and df[chart_config.x_column].nunique() > 6:
            warnings.append(
                f"Pie chart has {df[chart_config.x_column].nunique()} slices - "
                "consider a bar chart instead"
            )
            confidence -= 20

    # Line chart with no temporal ordering
    if chart_config.chart_type == ChartType.LINE:
        if chart_config.x_column:
            try:
                pd.to_datetime(df[chart_config.x_column])
            except (ValueError, TypeError):
                if not pd.api.types.is_numeric_dtype(df[chart_config.x_column]):
                    warnings.append("Line chart x-axis is not temporal or numeric - may be misleading")
                    confidence -= 15

    # Too few data points for trend
    if chart_config.chart_type == ChartType.LINE and len(df) < 3:
        warnings.append("Only {0} data points for trend line - insufficient for trend analysis".format(len(df)))
        confidence -= 20

    if confidence >= CRITIC_CONFIDENCE_THRESHOLD_WARN:
        status = ValidationStatus.VALIDATED
    elif confidence >= CRITIC_CONFIDENCE_THRESHOLD_REJECT:
        status = ValidationStatus.VALIDATED_WITH_WARNINGS
    else:
        status = ValidationStatus.REJECTED

    return ValidationReport(
        status=status,
        confidence=max(0, confidence),
        warnings=warnings,
        corrections_made=corrections,
    )


def llm_validate(
    question: str,
    sql_query: str,
    result_summary: str,
    schema: SemanticSchema,
) -> ValidationReport:
    """Use Gemini to validate the overall analysis quality."""
    client = genai.Client(api_key=GOOGLE_API_KEY)

    schema_desc = json.dumps(schema.model_dump(), indent=2, default=str)

    prompt = f"""{CRITIC_AGENT_SYSTEM_PROMPT}

SCHEMA: {schema_desc}

QUESTION: {question}
SQL QUERY: {sql_query}
RESULT SUMMARY: {result_summary}

Evaluate this analysis. Return a JSON object:
{{
  "confidence": <0-100>,
  "warnings": ["list of concerns"],
  "corrections": ["list of suggested fixes"]
}}

Return ONLY the JSON:"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )
        text = response.text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        data = json.loads(text)

        conf = data.get("confidence", 80)
        warns = data.get("warnings", [])
        corrs = data.get("corrections", [])

        if conf >= CRITIC_CONFIDENCE_THRESHOLD_WARN:
            status = ValidationStatus.VALIDATED
        elif conf >= CRITIC_CONFIDENCE_THRESHOLD_REJECT:
            status = ValidationStatus.VALIDATED_WITH_WARNINGS
        else:
            status = ValidationStatus.REJECTED

        return ValidationReport(
            status=status,
            confidence=conf,
            warnings=warns,
            corrections_made=corrs,
        )
    except Exception:
        return ValidationReport(
            status=ValidationStatus.VALIDATED_WITH_WARNINGS,
            confidence=60,
            warnings=["Could not perform LLM validation"],
        )
