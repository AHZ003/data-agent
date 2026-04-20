"""Coder Agent - SQL/Python code generation and execution."""

import json
import re
from google import genai
from google.genai import types as genai_types
import pandas as pd
from typing import Optional, Tuple

from config import (
    GOOGLE_API_KEY,
    MODEL_NAME,
    CODER_AGENT_SYSTEM_PROMPT,
    MAX_QUERY_ROWS,
    MAX_RETRY_ATTEMPTS,
)
from core.database import Database
from models.analysis_plan import SemanticSchema, CodeResult


# Error patterns that indicate the LLM provider is rate-limiting or the
# account quota is exhausted. These are terminal — retrying just burns
# more quota and makes the problem worse. The orchestrator treats any
# error string starting with this sentinel as fatal and skips retries.
QUOTA_SENTINEL = "QuotaExhausted"
_QUOTA_PATTERNS = ("RESOURCE_EXHAUSTED", "429", "quota", "rate limit", "rate_limit")


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p.lower() in msg for p in _QUOTA_PATTERNS)


def _build_sql_prompt(question: str, schema: SemanticSchema) -> str:
    """Build the prompt for SQL generation."""
    schema_desc = json.dumps(schema.model_dump(), indent=2, default=str)
    system = CODER_AGENT_SYSTEM_PROMPT.format(
        max_rows=MAX_QUERY_ROWS, table_name=schema.table_name
    )
    return f"""{system}

DATASET SCHEMA:
{schema_desc}

USER QUESTION: {question}

Write the SQL query:"""


_UNSUPPORTED_FUNCS = re.compile(
    r"\b(CORR|STDDEV|STDDEV_POP|STDDEV_SAMP|VARIANCE|VAR_POP|VAR_SAMP"
    r"|PERCENTILE_CONT|PERCENTILE_DISC|MEDIAN|STDEV"
    r"|REGEXP_LIKE|REGEXP_REPLACE|REGEXP_SUBSTR"
    r"|STRING_AGG|ARRAY_AGG|LISTAGG"
    r"|COVAR_POP|COVAR_SAMP|REGR_SLOPE|REGR_INTERCEPT)\s*\(",
    re.IGNORECASE,
)


def _check_unsupported_functions(sql: str) -> Optional[str]:
    """Return an error hint if SQL uses functions SQLite doesn't have."""
    m = _UNSUPPORTED_FUNCS.search(sql)
    if not m:
        return None
    func = m.group(1).upper()
    return (
        f"SQLite ERROR: function {func}() does not exist. "
        "You MUST compute this manually. For correlation use: "
        "(SUM(x*y) - SUM(x)*SUM(y)/COUNT(*)) / "
        "(SQRT((SUM(x*x) - SUM(x)*SUM(x)/COUNT(*)) * "
        "(SUM(y*y) - SUM(y)*SUM(y)/COUNT(*)))). "
        "For stddev use: SQRT(AVG(x*x) - AVG(x)*AVG(x)). "
        "Do NOT use table aliases. Query directly from the table."
    )


def _extract_sql(response_text: str) -> str:
    """Extract SQL query from LLM response."""
    # Try to extract from code block
    match = re.search(r"```(?:sqlite|sql)?\s*(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: strip any leading/trailing backtick fence (handles unclosed fences)
    cleaned = response_text.strip()
    cleaned = re.sub(r"^`{3,}(?:sqlite|sql)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?`{3,}\s*$", "", cleaned)
    # Remove any non-SQL preamble
    for prefix in ["Here is", "The SQL", "SQL:", "Query:"]:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip().lstrip(":")
    return cleaned.strip()


def _generate_sql(question: str, schema: SemanticSchema, error_context: str = "") -> str:
    """Generate SQL using Gemini API."""
    client = genai.Client(api_key=GOOGLE_API_KEY)

    prompt = _build_sql_prompt(question, schema)
    if error_context:
        prompt += f"\n\nPREVIOUS ATTEMPT FAILED WITH ERROR:\n{error_context}\nPlease fix the query."

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=1024,
        ),
    )

    return _extract_sql(response.text)


def _pandas_correlation(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Compute pairwise correlations using pandas — reliable fallback for SQLite."""
    if df is None:
        return None
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr().round(4)
    # Reshape into a readable table: col_a, col_b, correlation
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            rows.append({"Column_A": a, "Column_B": b, "Correlation": corr.loc[a, b]})
    return pd.DataFrame(rows).sort_values("Correlation", key=abs, ascending=False)


_CORRELATION_KEYWORDS = re.compile(
    r"\bcorrelat|\bpearson|\br[\s-]?value\b|relationship between.*numeric"
    r"|\bcovariance\b|\bspearman\b|\bassociation\b.*\bnumeric",
    re.IGNORECASE,
)


def execute_analysis(
    question: str,
    schema: SemanticSchema,
    db: Database,
    source_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[pd.DataFrame], CodeResult]:
    """
    Generate SQL from a question, execute it, and return results.

    Implements self-correction: if the query fails, retries up to MAX_RETRY_ATTEMPTS.
    Falls back to pandas for correlation queries that SQLite can't handle.
    """
    last_error = ""
    last_sql = ""
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            sql = _generate_sql(question, schema, error_context=last_error)
            last_sql = sql

            unsupported = _check_unsupported_functions(sql)
            if unsupported:
                last_error = unsupported
                continue

            result_df, error = db.execute_query(sql)

            if error:
                last_error = f"SQL: {sql}\nError: {error}"
                continue

            code_result = CodeResult(
                sql_query=sql,
                success=True,
                row_count=len(result_df) if result_df is not None else 0,
                columns=list(result_df.columns) if result_df is not None else [],
            )
            return result_df, code_result

        except Exception as e:
            if _is_quota_error(e):
                return None, CodeResult(
                    sql_query="",
                    success=False,
                    error=f"{QUOTA_SENTINEL}: {e}",
                )
            last_error = str(e)

    # SQL retries exhausted — try pandas fallback for correlation queries
    if _CORRELATION_KEYWORDS.search(question) and source_df is not None:
        corr_df = _pandas_correlation(source_df)
        if corr_df is not None and len(corr_df) > 0:
            return corr_df, CodeResult(
                sql_query="-- Computed via pandas df.corr() (SQLite lacks CORR())",
                success=True,
                row_count=len(corr_df),
                columns=list(corr_df.columns),
            )

    return None, CodeResult(
        sql_query=last_sql,
        success=False,
        error=f"Failed after {MAX_RETRY_ATTEMPTS} attempts. Last error: {last_error}",
    )


def generate_python_code(
    question: str, schema: SemanticSchema, context: str = ""
) -> str:
    """Generate Python analysis code for complex tasks that can't be done in SQL."""
    client = genai.Client(api_key=GOOGLE_API_KEY)

    schema_desc = json.dumps(schema.model_dump(), indent=2, default=str)
    prompt = f"""You are a Python data analysis expert. Given a dataset schema and question,
write Python code using pandas, numpy, scipy, or sklearn.

The DataFrame is available as the variable `df`.
Store the final result in a variable called `result`.

SCHEMA:
{schema_desc}

{f"ADDITIONAL CONTEXT: {context}" if context else ""}

QUESTION: {question}

Write ONLY the Python code:"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2048,
        ),
    )

    text = response.text
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
