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

_EDA_KEYWORDS = re.compile(
    r"\bdescribe\b.*\bdata\b|\bdata\b.*\bdescri|\bEDA\b|\bexploratory\b"
    r"|\boverview\b.*\bdata|\bdata\b.*\boverview"
    r"|\bsummar\w+\b.*\b(data|dataset|table)\b"
    r"|\breport\b.*\b(descri|data|dataset)"
    r"|\bdata\b.*\breport\b|\bprofile\b.*\b(data|dataset)"
    r"|\b(data|dataset)\b.*\bprofile",
    re.IGNORECASE,
)


def _pandas_eda(df: pd.DataFrame, schema: SemanticSchema) -> pd.DataFrame:
    """Comprehensive Exploratory Data Analysis using pandas + scipy."""
    import numpy as np
    from core.stats import iqr_outliers, skewness_check, correlation_test

    rows = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    # Dataset overview row
    rows.append({
        "Section": "Overview",
        "Metric": "Dataset Shape",
        "Value": f"{len(df):,} rows x {len(df.columns)} columns",
        "Detail": f"Numeric: {len(numeric_cols)}, Categorical: {len(cat_cols)}, Datetime: {len(date_cols)}",
    })
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    rows.append({
        "Section": "Overview",
        "Metric": "Missing Values",
        "Value": f"{total_missing:,} ({total_missing / total_cells * 100:.1f}%)",
        "Detail": "Total missing cells across all columns",
    })
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        rows.append({
            "Section": "Overview",
            "Metric": "Duplicate Rows",
            "Value": f"{dup_count:,} ({dup_count / len(df) * 100:.1f}%)",
            "Detail": "Exact duplicate rows",
        })

    # Numeric column stats
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        rows.append({
            "Section": "Numeric Stats",
            "Metric": col,
            "Value": f"mean={s.mean():.2f}, std={s.std():.2f}",
            "Detail": f"min={s.min():.2f}, 25%={s.quantile(.25):.2f}, "
                      f"median={s.median():.2f}, 75%={s.quantile(.75):.2f}, max={s.max():.2f}",
        })

        # Outlier analysis
        out = iqr_outliers(s)
        if out and out.n_outliers > 0:
            rows.append({
                "Section": "Outliers",
                "Metric": col,
                "Value": f"{out.n_outliers} outliers ({out.share * 100:.1f}%)",
                "Detail": f"Tukey fences: [{out.lower_fence:.2f}, {out.upper_fence:.2f}]",
            })

        # Skewness
        sk = skewness_check(s)
        if sk and sk.verdict != "symmetric":
            rows.append({
                "Section": "Distribution",
                "Metric": f"{col} skewness",
                "Value": f"{sk.skew:.3f} ({sk.verdict})",
                "Detail": f"p-value={sk.p_value:.4f}" if sk.p_value is not None else "n too small for test",
            })

    # Categorical column stats
    for col in cat_cols:
        n_unique = df[col].nunique()
        top = df[col].value_counts().head(3)
        top_str = ", ".join(f"{v} ({c})" for v, c in top.items())
        rows.append({
            "Section": "Categorical Stats",
            "Metric": col,
            "Value": f"{n_unique} unique values",
            "Detail": f"Top: {top_str}",
        })

    # Correlations between numeric columns
    if len(numeric_cols) >= 2:
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1:]:
                cr = correlation_test(df[a], df[b])
                if cr:
                    rows.append({
                        "Section": "Correlations",
                        "Metric": f"{a} vs {b}",
                        "Value": f"r={cr.r:.4f} ({cr.verdict})",
                        "Detail": f"p={cr.p_value:.6f}, n={cr.n}, significant={cr.significant}",
                    })

    # Missing value breakdown per column (only if any)
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    for col, count in missing.items():
        rows.append({
            "Section": "Missing Values",
            "Metric": str(col),
            "Value": f"{count:,} ({count / len(df) * 100:.1f}%)",
            "Detail": "Missing values in this column",
        })

    return pd.DataFrame(rows)


def execute_analysis(
    question: str,
    schema: SemanticSchema,
    db: Database,
    source_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[pd.DataFrame], CodeResult]:
    """
    Generate SQL from a question, execute it, and return results.

    Implements self-correction: if the query fails, retries up to MAX_RETRY_ATTEMPTS.
    Falls back to pandas for correlation and EDA queries that SQL handles poorly.
    """
    # EDA/describe questions are best handled by pandas, not SQL
    if _EDA_KEYWORDS.search(question) and source_df is not None:
        eda_df = _pandas_eda(source_df, schema)
        return eda_df, CodeResult(
            sql_query="-- Exploratory Data Analysis computed via pandas + scipy",
            success=True,
            row_count=len(eda_df),
            columns=list(eda_df.columns),
        )

    # Correlation questions: try pandas first (SQLite lacks CORR())
    if _CORRELATION_KEYWORDS.search(question) and source_df is not None:
        corr_df = _pandas_correlation(source_df)
        if corr_df is not None and len(corr_df) > 0:
            return corr_df, CodeResult(
                sql_query="-- Computed via pandas df.corr() + scipy pearsonr",
                success=True,
                row_count=len(corr_df),
                columns=list(corr_df.columns),
            )

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
