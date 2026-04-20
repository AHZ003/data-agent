"""LangGraph Workflow Orchestrator - Routes between agents and manages state."""

import time
import json
import pandas as pd
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END

from models.analysis_plan import (
    AnalysisPlan,
    AnalysisType,
    SemanticSchema,
    AgentLogEntry,
    ValidationStatus,
)
from models.chart_config import ChartConfig
from agents import (
    schema_agent,
    planner_agent,
    coder_agent,
    visualizer_agent,
    predictor_agent,
    critic_agent,
    storyteller_agent,
)
from core.database import Database
from core import tracing
from config import MODEL_NAME


# --- State Definition ---

class AgentState(TypedDict):
    """State passed between agents in the graph."""
    question: str
    schema: Optional[dict]
    plan: Optional[dict]
    sql_query: Optional[str]
    result_df: Optional[Any]  # pd.DataFrame serialized
    chart: Optional[Any]  # plotly Figure
    chart_config: Optional[dict]
    validation: Optional[dict]
    prediction: Optional[dict]
    prediction_chart: Optional[Any]
    narrative: Optional[str]
    error: Optional[str]
    retry_count: int
    agent_log: List[dict]
    skip_storyteller: Optional[bool]
    # References (not serialized in state)
    db: Optional[Any]
    df: Optional[Any]


# --- Agent Node Functions ---

def planner_node(state: AgentState) -> AgentState:
    """Run the Planner Agent to create an analysis plan."""
    start = time.time()
    schema = SemanticSchema(**state["schema"])
    with tracing.span("planner", model=MODEL_NAME) as _t:
        plan = planner_agent.create_analysis_plan(state["question"], schema)
        _t.add_metadata(
            analysis_types=[t.value for t in plan.analysis_types],
            n_steps=len(plan.steps),
        )
    duration = time.time() - start

    log_entry = AgentLogEntry(
        agent_name="Planner",
        task="Create analysis plan",
        input_summary=f"Question: {state['question']}",
        output_summary=f"Types: {[t.value for t in plan.analysis_types]}, Steps: {len(plan.steps)}",
        duration_seconds=round(duration, 2),
    )
    state["plan"] = plan.model_dump()
    state["agent_log"].append(log_entry.model_dump())
    return state


def coder_node(state: AgentState) -> AgentState:
    """Run the Coder Agent to generate and execute SQL.

    Increments `retry_count` on every entry so the conditional edge
    from the critic can actually bound retries (see should_retry).
    Before this fix the counter was read but never written, so a
    persistently-rejecting critic produced an infinite loop. Captured
    in docs/postmortems/2026-04-13_eval_findings.md.
    """
    start = time.time()
    schema = SemanticSchema(**state["schema"])
    db = state["db"]
    state["retry_count"] = state.get("retry_count", 0) + 1

    with tracing.span("coder", model=MODEL_NAME) as _t:
        result_df, code_result = coder_agent.execute_analysis(
            state["question"], schema, db, source_df=state.get("df"),
        )
        _t.add_metadata(
            sql=(code_result.sql_query or "")[:300],
            row_count=code_result.row_count,
            success=code_result.success,
        )
    duration = time.time() - start

    log_entry = AgentLogEntry(
        agent_name="Coder",
        task="Generate and execute SQL",
        input_summary=f"Question: {state['question']}",
        output_summary=f"SQL: {code_result.sql_query[:100]}... | Rows: {code_result.row_count}",
        duration_seconds=round(duration, 2),
        status="success" if code_result.success else "error",
    )

    state["sql_query"] = code_result.sql_query
    state["result_df"] = result_df
    state["error"] = code_result.error
    state["agent_log"].append(log_entry.model_dump())
    return state


def critic_node(state: AgentState) -> AgentState:
    """Run the Critic Agent to validate results."""
    start = time.time()
    schema = SemanticSchema(**state["schema"])
    result_df = state.get("result_df")

    # Validate SQL results
    with tracing.span("critic", model=MODEL_NAME) as _t:
        validation = critic_agent.validate_sql_result(
            result_df, state.get("sql_query", ""), state["question"], schema
        )
        if state.get("chart_config") and result_df is not None:
            chart_config = ChartConfig(**state["chart_config"])
            chart_validation = critic_agent.validate_chart(result_df, chart_config)
            validation.warnings.extend(chart_validation.warnings)
            validation.corrections_made.extend(chart_validation.corrections_made)
            validation.confidence = min(validation.confidence, chart_validation.confidence)
        _t.add_metadata(
            status=validation.status.value,
            confidence=validation.confidence,
            n_warnings=len(validation.warnings),
        )

    duration = time.time() - start

    log_entry = AgentLogEntry(
        agent_name="Critic",
        task="Validate analysis",
        input_summary=f"Validating SQL result ({len(result_df) if result_df is not None else 0} rows)",
        output_summary=f"Status: {validation.status.value}, Confidence: {validation.confidence}",
        duration_seconds=round(duration, 2),
    )

    state["validation"] = validation.model_dump()
    state["agent_log"].append(log_entry.model_dump())
    return state


def visualizer_node(state: AgentState) -> AgentState:
    """Run the Visualizer Agent to generate charts."""
    start = time.time()
    result_df = state.get("result_df")

    with tracing.span("visualizer", model=MODEL_NAME) as _t:
        if result_df is not None and len(result_df) > 0:
            chart_config = visualizer_agent.get_chart_config(result_df, state["question"])
            chart = visualizer_agent.generate_chart(result_df, state["question"], chart_config)
            state["chart"] = chart
            state["chart_config"] = chart_config.model_dump()
            _t.add_metadata(chart_type=getattr(chart_config.chart_type, "value", str(chart_config.chart_type)))
        else:
            state["chart"] = None
            state["chart_config"] = None
            _t.add_metadata(chart_type="none")

    duration = time.time() - start

    log_entry = AgentLogEntry(
        agent_name="Visualizer",
        task="Generate chart",
        input_summary=f"Data: {len(result_df) if result_df is not None else 0} rows",
        output_summary=f"Chart type: {state['chart_config']['chart_type'] if state.get('chart_config') else 'none'}",
        duration_seconds=round(duration, 2),
    )
    state["agent_log"].append(log_entry.model_dump())
    return state


def predictor_node(state: AgentState) -> AgentState:
    """Run the Predictor Agent for ML tasks."""
    start = time.time()
    plan = state.get("plan", {})
    analysis_types = plan.get("analysis_types", [])
    df = state.get("df")
    schema = SemanticSchema(**state["schema"])

    if df is None:
        state["prediction"] = {"error": "No data available for prediction"}
        return state

    try:
        if "prediction" in analysis_types:
            # Find time and value columns
            time_cols = [c for c in schema.columns if c.role == "time_axis"]
            measure_cols = [c for c in schema.columns if c.role == "measure"]
            if time_cols and measure_cols:
                forecast_df, fig, metrics = predictor_agent.forecast_time_series(
                    df, time_cols[0].name, measure_cols[0].name
                )
                state["prediction"] = metrics
                state["prediction_chart"] = fig
            else:
                state["prediction"] = {"error": "No suitable time series columns found"}

        elif "segmentation" in analysis_types:
            clustered_df, fig, metrics = predictor_agent.cluster_data(df)
            state["prediction"] = metrics
            state["prediction_chart"] = fig
            state["result_df"] = clustered_df

        elif "anomaly" in analysis_types:
            anomaly_df, fig, metrics = predictor_agent.detect_anomalies(df)
            state["prediction"] = metrics
            state["prediction_chart"] = fig
            state["result_df"] = anomaly_df

    except Exception as e:
        state["prediction"] = {"error": str(e)}

    duration = time.time() - start

    log_entry = AgentLogEntry(
        agent_name="Predictor",
        task="ML analysis",
        input_summary=f"Types: {analysis_types}",
        output_summary=json.dumps(state.get("prediction", {}), default=str)[:200],
        duration_seconds=round(duration, 2),
    )
    state["agent_log"].append(log_entry.model_dump())
    return state


def storyteller_node(state: AgentState) -> AgentState:
    """Run the Storyteller Agent to generate narrative.

    Callers that want to stream the narrative themselves (e.g. the chat
    UI, which uses storyteller_agent.stream_narrative for real
    token-by-token output) can set state["skip_storyteller"]=True to
    avoid paying 2× LLM cost for the same text.
    """
    if state.get("skip_storyteller"):
        state["narrative"] = ""
        return state

    start = time.time()
    result_df = state.get("result_df")
    result_summary = ""
    if result_df is not None:
        result_summary = result_df.head(20).to_string()

    validation = state.get("validation", {})
    warnings = validation.get("warnings", [])

    with tracing.span("storyteller", model=MODEL_NAME) as _t:
        narrative = storyteller_agent.generate_narrative(
            question=state["question"],
            sql_query=state.get("sql_query", ""),
            result_summary=result_summary,
            chart_description=state.get("chart_config", {}).get("chart_type", ""),
            validation_warnings=warnings,
            prediction_info=state.get("prediction"),
        )
        _t.add_metadata(narrative_length=len(narrative or ""))
    duration = time.time() - start

    log_entry = AgentLogEntry(
        agent_name="Storyteller",
        task="Generate narrative",
        input_summary=f"Question: {state['question']}",
        output_summary=narrative[:200],
        duration_seconds=round(duration, 2),
    )
    state["narrative"] = narrative
    state["agent_log"].append(log_entry.model_dump())
    return state


# --- Routing Logic ---

def should_predict(state: AgentState) -> str:
    """Determine if prediction is needed."""
    plan = state.get("plan", {})
    analysis_types = plan.get("analysis_types", [])
    prediction_types = {"prediction", "segmentation", "anomaly"}
    if prediction_types.intersection(set(analysis_types)):
        return "predictor"
    return "storyteller"


def should_retry(state: AgentState) -> str:
    """Determine if we should retry after critic rejection.

    Retries are bounded by `retry_count` (incremented in coder_node)
    and are skipped entirely when the coder hit a terminal error such
    as an LLM quota exhaustion — retrying a 429 storm just amplifies
    the outage.
    """
    error = state.get("error") or ""
    if "QuotaExhausted" in error:
        return "continue"

    validation = state.get("validation", {})
    status = validation.get("status", "validated")
    retry_count = state.get("retry_count", 0)

    if status == "rejected" and retry_count < 3:
        return "retry"
    return "continue"


# --- Build the Graph ---

def build_graph() -> StateGraph:
    """Build the LangGraph agent workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("critic", critic_node)
    graph.add_node("visualizer", visualizer_node)
    graph.add_node("predictor", predictor_node)
    graph.add_node("storyteller", storyteller_node)

    # Define edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "visualizer")
    graph.add_edge("visualizer", "critic")

    # Conditional: retry or continue after critic
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"retry": "coder", "continue": "decide_predict"},
    )

    # Add a decision node for prediction routing
    graph.add_node("decide_predict", lambda state: state)
    graph.add_conditional_edges(
        "decide_predict",
        should_predict,
        {"predictor": "predictor", "storyteller": "storyteller"},
    )
    graph.add_edge("predictor", "storyteller")
    graph.add_edge("storyteller", END)

    return graph


def run_analysis(
    question: str,
    schema: SemanticSchema,
    db: Database,
    df: pd.DataFrame,
) -> AgentState:
    """
    Run the full agent pipeline for a question.

    Returns the final state with all results.
    """
    graph = build_graph()
    app = graph.compile()

    initial_state: AgentState = {
        "question": question,
        "schema": schema.model_dump(),
        "plan": None,
        "sql_query": None,
        "result_df": None,
        "chart": None,
        "chart_config": None,
        "validation": None,
        "prediction": None,
        "prediction_chart": None,
        "narrative": None,
        "error": None,
        "retry_count": 0,
        "agent_log": [],
        "db": db,
        "df": df,
    }

    run_id = tracing.new_run(question, dataset=schema.table_name)
    try:
        result = app.invoke(initial_state)
        tracing.end_run(status="ok")
        if isinstance(result, dict):
            result["run_id"] = run_id
        return result
    except Exception as e:
        tracing.end_run(status="error", error=str(e))
        raise


def run_analysis_stream(
    question: str,
    schema: SemanticSchema,
    db: Database,
    df: pd.DataFrame,
    skip_storyteller: bool = False,
):
    """
    Streaming version of run_analysis.

    Yields (node_name, merged_state) after each graph node executes, then
    a final ("done", final_state) event. Callers can use this to drive a
    live progress UI (e.g. st.status).
    """
    graph = build_graph()
    app = graph.compile()

    initial_state: AgentState = {
        "question": question,
        "schema": schema.model_dump(),
        "plan": None,
        "sql_query": None,
        "result_df": None,
        "chart": None,
        "chart_config": None,
        "validation": None,
        "prediction": None,
        "prediction_chart": None,
        "narrative": None,
        "error": None,
        "retry_count": 0,
        "agent_log": [],
        "db": db,
        "df": df,
        "skip_storyteller": skip_storyteller,
    }

    run_id = tracing.new_run(question, dataset=schema.table_name)
    merged: dict = dict(initial_state)
    merged["run_id"] = run_id
    try:
        for chunk in app.stream(initial_state):
            for node_name, node_state in chunk.items():
                if isinstance(node_state, dict):
                    merged.update(node_state)
                yield node_name, merged
        tracing.end_run(status="ok")
    except Exception as e:
        tracing.end_run(status="error", error=str(e))
        raise

    yield "done", merged
