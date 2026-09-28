"""Planner Agent - Analysis plan generation and question classification."""

import json
import re
from typing import List

from config import PLANNER_AGENT_SYSTEM_PROMPT, model_for
from core import llm
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
  ],
  "clarifying_question": null
}}

Valid agents: coder, visualizer, predictor, critic, storyteller
Valid analysis_types: descriptive, trend, comparison, correlation, prediction, segmentation, anomaly, report

Return ONLY the JSON object:"""

    try:
        text = llm.generate(prompt, model=model_for("planner"), temperature=0.0,
                            max_output_tokens=1024).strip()
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

        clarify = data.get("clarifying_question")
        return AnalysisPlan(
            question=question,
            analysis_types=analysis_types,
            steps=steps,
            clarifying_question=clarify.strip() if isinstance(clarify, str) and clarify.strip() else None,
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
