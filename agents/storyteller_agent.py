"""Storyteller Agent - Narrative generation and report assembly."""

import json
import re
from google import genai
from google.genai import types as genai_types
from typing import Iterator, List, Dict, Any, Optional

from config import MODEL_NAME, STORYTELLER_AGENT_SYSTEM_PROMPT
from core import injection, tracing
from core.llm import current_api_key
from models.report import AnalysisReport, ReportSection


def _build_narrative_prompt(
    question: str,
    sql_query: str,
    result_summary: str,
    chart_description: str = "",
    validation_warnings: Optional[List[str]] = None,
    prediction_info: Optional[Dict[str, Any]] = None,
) -> str:
    if injection.enabled():
        # String-level defense for callers that pass raw text (summarize_result
        # already neutralizes at the cell level).
        result_summary = injection._RX.sub(injection.PLACEHOLDER, result_summary or "")
    context_parts = [
        f"Question: {question}",
        f"SQL Query: {sql_query}",
        "Results:\n" + injection.wrap("query_results", result_summary),
    ]
    if chart_description:
        context_parts.append(f"Chart: {chart_description}")
    if validation_warnings:
        context_parts.append(f"Warnings: {', '.join(validation_warnings)}")
    if prediction_info:
        context_parts.append(f"Prediction: {json.dumps(prediction_info, default=str)}")

    notice = f"\n{injection.DATA_NOTICE}\n" if injection.enabled() else ""
    return f"""{STORYTELLER_AGENT_SYSTEM_PROMPT}
{notice}
ANALYSIS CONTEXT:
{chr(10).join(context_parts)}

Write the narrative now. Lead with the key finding, include specific numbers, and end with an actionable takeaway."""


def summarize_result(df, pii_report=None, max_rows: int = 20) -> str:
    """The result table as the Storyteller may see it.

    PII columns are masked, instruction-like cells neutralized, and in
    local-only mode only the shape and numeric summaries are shared.
    """
    from core import pii

    if df is None:
        return ""
    if pii.local_only():
        num = df.select_dtypes("number")
        desc = num.describe().round(4).to_string() if not num.empty else "(no numeric columns)"
        return f"{len(df)} rows; columns: {', '.join(map(str, df.columns))}\nNumeric summary:\n{desc}"
    safe = injection.sanitize_frame(df.head(max_rows))
    if pii_report is not None:
        safe = pii.mask_frame(safe, pii_report)
    return safe.to_string()


def generate_narrative(
    question: str,
    sql_query: str,
    result_summary: str,
    chart_description: str = "",
    validation_warnings: Optional[List[str]] = None,
    prediction_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a plain-English narrative for a single analysis step (blocking)."""
    client = genai.Client(api_key=current_api_key())
    prompt = _build_narrative_prompt(
        question, sql_query, result_summary,
        chart_description, validation_warnings, prediction_info,
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=512,
        ),
    )
    tracing.record_usage(response, MODEL_NAME)
    return response.text.strip()


def stream_narrative(
    question: str,
    sql_query: str,
    result_summary: str,
    chart_description: str = "",
    validation_warnings: Optional[List[str]] = None,
    prediction_info: Optional[Dict[str, Any]] = None,
) -> Iterator[str]:
    """Yield narrative text in real time as Gemini generates it.

    This is the actual-SDK-streaming path — not the fake-time-sleep
    word-chunker. Callers like `st.write_stream` can consume this
    iterator directly.
    """
    client = genai.Client(api_key=current_api_key())
    prompt = _build_narrative_prompt(
        question, sql_query, result_summary,
        chart_description, validation_warnings, prediction_info,
    )
    last_chunk = None
    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        ):
            last_chunk = chunk
            text = getattr(chunk, "text", None)
            if text:
                yield text
        # Streamed usage metadata is cumulative; the last chunk has the total.
        tracing.record_usage(last_chunk, MODEL_NAME)
    except Exception as e:
        yield f"\n\n_Narrative generation encountered an error: {e}_"


def generate_full_report(
    session_data: Dict[str, Any],
) -> AnalysisReport:
    """
    Generate a comprehensive report from all analysis in the session.

    session_data should contain:
    - schema_summary: str
    - analyses: list of {question, sql, result_summary, narrative, warnings}
    - predictions: list of prediction results (optional)
    """
    client = genai.Client(api_key=current_api_key())

    analyses = session_data.get("analyses", [])
    predictions = session_data.get("predictions", [])
    schema_summary = session_data.get("schema_summary", "")

    # Build analysis summaries for context
    analysis_text = ""
    for i, a in enumerate(analyses, 1):
        analysis_text += f"\n--- Analysis {i} ---\n"
        analysis_text += f"Question: {a.get('question', '')}\n"
        analysis_text += f"Result: {a.get('result_summary', '')}\n"
        analysis_text += f"Narrative: {a.get('narrative', '')}\n"
        if a.get("warnings"):
            analysis_text += f"Warnings: {', '.join(a['warnings'])}\n"

    prediction_text = ""
    for p in predictions:
        prediction_text += f"\nPrediction: {json.dumps(p, default=str)}\n"

    prompt = f"""Generate a comprehensive analysis report.

DATA OVERVIEW:
{schema_summary}

ANALYSES PERFORMED:
{analysis_text}

{f"PREDICTIONS:{prediction_text}" if prediction_text else ""}

Return a JSON object with this structure:
{{
  "title": "Analysis Report: <dataset topic>",
  "executive_summary": "2-3 sentence overview of key findings",
  "data_overview": "Description of the dataset and quality notes",
  "findings": [
    {{"title": "Finding title", "content": "Detailed finding narrative"}}
  ],
  "predictions": "Summary of predictions (or null)",
  "caveats": ["list of limitations and caveats"],
  "next_questions": ["list of suggested follow-up questions"]
}}

Return ONLY the JSON:"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )
        tracing.record_usage(response, MODEL_NAME)
        text = response.text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        data = json.loads(text)

        findings = [
            ReportSection(title=f.get("title", ""), content=f.get("content", ""))
            for f in data.get("findings", [])
        ]

        return AnalysisReport(
            title=data.get("title", "Analysis Report"),
            executive_summary=data.get("executive_summary", ""),
            data_overview=data.get("data_overview", ""),
            findings=findings,
            predictions=data.get("predictions"),
            caveats=data.get("caveats", []),
            next_questions=data.get("next_questions", []),
        )
    except Exception as e:
        return AnalysisReport(
            title="Analysis Report",
            executive_summary="Report generation encountered an error.",
            data_overview=schema_summary,
            findings=[
                ReportSection(title=a.get("question", ""), content=a.get("narrative", ""))
                for a in analyses
            ],
            caveats=[f"Report generation error: {str(e)}"],
            next_questions=["Try regenerating the report"],
        )


def report_to_markdown(report: AnalysisReport) -> str:
    """Convert an AnalysisReport to Markdown format."""
    lines = [
        f"# {report.title}",
        "",
        "## Executive Summary",
        report.executive_summary,
        "",
        "## Data Overview",
        report.data_overview,
        "",
        "## Key Findings",
    ]

    for finding in report.findings:
        lines.append(f"\n### {finding.title}")
        lines.append(finding.content)

    if report.predictions:
        lines.append("\n## Predictions")
        lines.append(report.predictions)

    if report.caveats:
        lines.append("\n## Caveats and Limitations")
        for caveat in report.caveats:
            lines.append(f"- {caveat}")

    if report.next_questions:
        lines.append("\n## Suggested Next Questions")
        for q in report.next_questions:
            lines.append(f"- {q}")

    return "\n".join(lines)
