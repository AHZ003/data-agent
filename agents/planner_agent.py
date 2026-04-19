"""Planner Agent - Analysis plan generation and question classification."""

import json
import re
from google import genai
from google.genai import types as genai_types
from typing import List

from config import GOOGLE_API_KEY, MODEL_NAME, PLANNER_AGENT_SYSTEM_PROMPT
from models.analysis_plan import (
    AnalysisPlan,
    AnalysisType,
    PlanStep,
    SemanticSchema,
)


def create_analysis_plan(
    question: str, schema: SemanticSchema
) -> AnalysisPlan:
    """
    Classify a question and create a step-by-step analysis plan.
    """
    client = genai.Client(api_key=GOOGLE_API_KEY)

    schema_summary = json.dumps(schema.model_dump(), indent=2, default=str)

    prompt = f"""{PLANNER_AGENT_SYSTEM_PROMPT}

DATASET SCHEMA:
{schema_summary}

USER QUESTION: {question}

Return a JSON object with this exact structure:
{{
  "question": "<the question>",
  "analysis_types": ["descriptive", "trend", ...],
  "steps": [
    {{"agent": "coder", "task": "description of what SQL to write"}},
    {{"agent": "visualizer", "task": "description of chart to create"}},
    {{"agent": "predictor", "task": "description of prediction to make"}},
    {{"agent": "storyteller", "task": "description of narrative to generate"}}
  ]
}}

Valid agents: coder, visualizer, predictor, critic, storyteller
Valid analysis_types: descriptive, trend, comparison, correlation, prediction, segmentation, anomaly, report

Return ONLY the JSON object:"""

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
    except Exception:
        return _fallback_plan(question)

    # Parse JSON from response
    try:
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        data = json.loads(text)
    except json.JSONDecodeError:
        return _fallback_plan(question)

    # Build the plan from parsed data
    try:
        analysis_types = []
        for at in data.get("analysis_types", ["descriptive"]):
            try:
                analysis_types.append(AnalysisType(at))
            except ValueError:
                pass
        if not analysis_types:
            analysis_types = [AnalysisType.DESCRIPTIVE]

        steps = []
        for step_data in data.get("steps", []):
            steps.append(
                PlanStep(
                    agent=step_data.get("agent", "coder"),
                    task=step_data.get("task", ""),
                )
            )
        if not steps:
            steps = [PlanStep(agent="coder", task=f"Answer: {question}")]

        return AnalysisPlan(
            question=question,
            analysis_types=analysis_types,
            steps=steps,
        )
    except Exception:
        return _fallback_plan(question)


def _fallback_plan(question: str) -> AnalysisPlan:
    """Create a simple fallback plan when LLM parsing fails."""
    return AnalysisPlan(
        question=question,
        analysis_types=[AnalysisType.DESCRIPTIVE],
        steps=[
            PlanStep(agent="coder", task=f"Write SQL to answer: {question}"),
            PlanStep(agent="visualizer", task="Auto-generate appropriate chart"),
            PlanStep(agent="storyteller", task="Summarize the findings"),
        ],
    )
