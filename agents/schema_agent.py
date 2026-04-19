"""Schema Agent - Data profiling and semantic schema detection."""

import json
import pandas as pd
import numpy as np
from google import genai
from google.genai import types as genai_types
from typing import List

from config import (
    GOOGLE_API_KEY,
    MODEL_NAME,
    SCHEMA_AGENT_SYSTEM_PROMPT,
    SUGGESTED_QUESTIONS_PROMPT,
    DEFAULT_TABLE_NAME,
)
from models.analysis_plan import ColumnProfile, ColumnRole, SemanticSchema

_SUGGESTION_CACHE: dict = {}


def _suggestion_cache_key(schema: SemanticSchema) -> tuple:
    """Signature for cache lookup — stable across identical datasets."""
    return (
        schema.table_name,
        schema.row_count,
        tuple((c.name, c.role.value, c.dtype) for c in schema.columns),
    )


def _detect_column_role(series: pd.Series, col_name: str) -> ColumnRole:
    """Detect the semantic role of a column."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnRole.TIME_AXIS
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.nunique()
        # Only classify as identifier if integer-like with all unique values
        if (nunique == len(series) and nunique > 20
                and pd.api.types.is_integer_dtype(series)):
            return ColumnRole.IDENTIFIER
        return ColumnRole.MEASURE
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        # Try parsing as datetime
        try:
            pd.to_datetime(series.dropna().head(20))
            return ColumnRole.TIME_AXIS
        except (ValueError, TypeError):
            pass
        nunique = series.nunique()
        if nunique / max(len(series), 1) > 0.8:
            return ColumnRole.IDENTIFIER
        if nunique <= 50:
            return ColumnRole.DIMENSION
        return ColumnRole.TEXT
    return ColumnRole.DIMENSION


def _profile_column(series: pd.Series, col_name: str) -> ColumnProfile:
    """Generate a profile for a single column."""
    role = _detect_column_role(series, col_name)
    null_count = int(series.isnull().sum())
    null_pct = round(null_count / max(len(series), 1) * 100, 2)
    unique_count = int(series.nunique())

    profile = ColumnProfile(
        name=col_name,
        dtype=str(series.dtype),
        role=role,
        null_count=null_count,
        null_percentage=null_pct,
        unique_count=unique_count,
    )

    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) > 0:
            profile.min_val = round(float(clean.min()), 4)
            profile.max_val = round(float(clean.max()), 4)
            profile.mean_val = round(float(clean.mean()), 4)
            profile.median_val = round(float(clean.median()), 4)
            profile.std_val = round(float(clean.std()), 4)

    if role == ColumnRole.DIMENSION:
        top = series.value_counts().head(5).index.tolist()
        profile.top_values = [str(v) for v in top]

    if role == ColumnRole.TIME_AXIS:
        try:
            dates = pd.to_datetime(series.dropna())
            profile.date_range = f"{dates.min()} to {dates.max()}"
            if len(dates) > 1:
                diffs = dates.sort_values().diff().dropna()
                median_diff = diffs.median()
                if median_diff <= pd.Timedelta(days=1):
                    profile.date_frequency = "daily"
                elif median_diff <= pd.Timedelta(days=7):
                    profile.date_frequency = "weekly"
                elif median_diff <= pd.Timedelta(days=31):
                    profile.date_frequency = "monthly"
                else:
                    profile.date_frequency = "yearly"
        except (ValueError, TypeError):
            pass

    return profile


def _suggest_analyses(columns: List[ColumnProfile]) -> List[str]:
    """Suggest analysis types based on column profiles."""
    suggestions = []
    has_time = any(c.role == ColumnRole.TIME_AXIS for c in columns)
    has_measure = any(c.role == ColumnRole.MEASURE for c in columns)
    has_dimension = any(c.role == ColumnRole.DIMENSION for c in columns)
    measure_count = sum(1 for c in columns if c.role == ColumnRole.MEASURE)

    if has_time and has_measure:
        suggestions.append("time series trend analysis")
    if has_dimension and has_measure:
        suggestions.append("category comparison")
    if measure_count >= 2:
        suggestions.append("correlation analysis")
        suggestions.append("clustering/segmentation")
    if has_measure:
        suggestions.append("anomaly detection")
    if has_time and has_measure:
        suggestions.append("forecasting")

    return suggestions


def _generate_suggested_questions(
    schema: SemanticSchema,
) -> List[str]:
    """Use Gemini to generate suggested questions for the dataset."""
    client = genai.Client(api_key=GOOGLE_API_KEY)

    schema_summary = json.dumps(schema.model_dump(), indent=2, default=str)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"Dataset profile:\n{schema_summary}\n\n{SUGGESTED_QUESTIONS_PROMPT}",
            config=genai_types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
            ),
        )
        text = response.text.strip()
        # Parse JSON array from response
        if "[" in text:
            json_str = text[text.index("[") : text.rindex("]") + 1]
            return json.loads(json_str)
    except Exception:
        pass

    return [
        "What are the top 10 values by the main metric?",
        "How does the data trend over time?",
        "What is the distribution of the main categories?",
        "Are there any outliers in the data?",
        "What correlations exist between numeric columns?",
    ]


def profile_dataframe(
    df: pd.DataFrame, table_name: str = DEFAULT_TABLE_NAME
) -> SemanticSchema:
    """
    Profile a DataFrame and generate a semantic schema.

    This is the main entry point for the Schema Agent.
    """
    # Auto-detect and convert datetime columns
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass

    columns = [_profile_column(df[col], col) for col in df.columns]
    analyses = _suggest_analyses(columns)

    schema = SemanticSchema(
        table_name=table_name,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        suggested_analyses=analyses,
    )

    # Generate AI-powered suggested questions (cached by dataset signature)
    key = _suggestion_cache_key(schema)
    cached = _SUGGESTION_CACHE.get(key)
    if cached is not None:
        schema.suggested_questions = cached
    else:
        schema.suggested_questions = _generate_suggested_questions(schema)
        _SUGGESTION_CACHE[key] = schema.suggested_questions

    return schema
