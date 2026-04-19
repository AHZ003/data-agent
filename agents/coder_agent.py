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


def execute_analysis(
    question: str,
    schema: SemanticSchema,
    db: Database,
) -> Tuple[Optional[pd.DataFrame], CodeResult]:
    """
    Generate SQL from a question, execute it, and return results.

    Implements self-correction: if the query fails, retries up to MAX_RETRY_ATTEMPTS.
    """
    last_error = ""
    last_sql = ""
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            sql = _generate_sql(question, schema, error_context=last_error)
            last_sql = sql
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
            # Quota / rate-limit errors are terminal. Retrying makes it
            # worse and can wedge the orchestrator in an infinite loop
            # (see docs/postmortems/2026-04-13_eval_findings.md).
            if _is_quota_error(e):
                return None, CodeResult(
                    sql_query="",
                    success=False,
                    error=f"{QUOTA_SENTINEL}: {e}",
                )
            last_error = str(e)

    # All retries exhausted — preserve the last SQL we tried instead
    # of overwriting it with the error string.
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
