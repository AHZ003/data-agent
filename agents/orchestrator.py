"""LangGraph Workflow Orchestrator - Routes between agents and manages state."""

import time
import json
import pandas as pd
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

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
from core import resources, tracing
from core.semantic_cache import default_cache, schema_version
from config import MODEL_NAME


# --- State Definition ---

class AgentState(TypedDict):
    """State passed between agents in the graph."""
    question: str
    display_question: Optional[str]  # Original user prompt (no context prefix) for chart titles
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
    # The engine and source DataFrame live in core.resources, not in state,
    # so state stays serializable for the checkpointer.
    datasource_id: Optional[str]
    # Earlier turns of this conversation: [{question, sql, result_preview}].
    history: List[dict]
    cache_hit: Optional[bool]


def _resources(state) -> tuple:
    """(engine, source DataFrame) for this run.

    A direct `db` (legacy callers and unit tests) wins; otherwise the
    registered datasource_id is resolved.
    """
    if state.get("db") is not None:
        return state["db"], state.get("df")
    return resources.get(state["datasource_id"])


def _sample_primary_table(db, n: int = 50_000) -> pd.DataFrame:
    q = '"' + db.table_name.replace('"', '""') + '"'
    df, err = db.execute_query(f"SELECT * FROM {q}", max_rows=n)
    return df if err is None else pd.DataFrame()


def _with_history(question: str, history: list) -> str:
    """Prefix a follow-up with the conversation so far ("now by region")."""
    if not history:
        return question
    turns = "\n".join(
        f"{i}. Q: {h['question']}\n   SQL: {h.get('sql') or '-'}\n   Result (first rows): {h.get('result_preview') or '-'}"
        for i, h in enumerate(history[-3:], 1)
    )
    return (f"CONVERSATION SO FAR (use it to resolve references like 'that', 'those', 'by region'):\n"
            f"{turns}\n\nCURRENT QUESTION: {question}")


# --- Agent Node Functions ---

def schema_node(state: AgentState) -> AgentState:
    """Run the Schema Agent: profile the primary table's columns.

    Callers that already profiled the dataset (the UI does so on upload
    to render the profile) pass it in and the node reuses it; otherwise
    it profiles `df` here. No LLM call either way.
    """
    start = time.time()
    reused = state.get("schema") is not None
    with tracing.span("schema") as _t:
        if not reused:
            db, df = _resources(state)
            if df is None:  # file-backed engine: profile a sample of the primary table
                df = _sample_primary_table(db)
            schema = schema_agent.profile_dataframe(
                df, db.table_name, suggest_questions=False,
            )
            state["schema"] = schema.model_dump()
        _t.add_metadata(reused=reused, n_columns=len(state["schema"]["columns"]))
    duration = time.time() - start

    log_entry = AgentLogEntry(
        agent_name="Schema",
        task="Profile columns",
        input_summary=f"Table: {state['schema']['table_name']}",
        output_summary=("Reused existing profile" if reused else "Profiled")
        + f" ({state['schema']['column_count']} columns)",
        duration_seconds=round(duration, 2),
    )
    state["agent_log"].append(log_entry.model_dump())
    return state


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
    db, df = _resources(state)
    state["retry_count"] = state.get("retry_count", 0) + 1
    history = state.get("history") or []
    cache = default_cache() if not history else None  # follow-ups depend on context
    version = schema_version(db.datasource()) if cache else None

    with tracing.span("coder", model=MODEL_NAME) as _t:
        result_df, code_result, hit = None, None, False
        cached_sql = cache.lookup(state["question"], version) if cache else None
        if cached_sql:
            result_df, err = db.execute_query(cached_sql)
            if err is None:
                hit = True
                code_result = coder_agent.CodeResult(
                    sql_query=cached_sql, success=True, row_count=len(result_df),
                    columns=list(result_df.columns), attempts=0,
                )
        if code_result is None:
            with tracing.usage_scope() as usage:
                result_df, code_result = coder_agent.execute_analysis(
                    _with_history(state["question"], history), schema, db, source_df=df,
                )
            if cache and code_result.success and code_result.sql_query.lstrip().lower().startswith(("select", "with")):
                cache.store(state["question"], version, code_result.sql_query, usage.cost_usd)
        _t.add_metadata(
            sql=(code_result.sql_query or "")[:300],
            row_count=code_result.row_count,
            success=code_result.success,
            cache_hit=hit,
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
    state["cache_hit"] = hit
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
    # Use display_question (original user prompt) for chart titles to avoid
    # showing the "Earlier questions in this conversation:..." prefix.
    chart_q = state.get("display_question") or state["question"]

    with tracing.span("visualizer", model=MODEL_NAME) as _t:
        if result_df is not None and len(result_df) > 0:
            chart_config = visualizer_agent.get_chart_config(result_df, chart_q)
            chart = visualizer_agent.generate_chart(result_df, chart_q, chart_config)
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
    _, df = _resources(state)
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
    and are skipped entirely when the coder hit a terminal error (LLM
    quota exhaustion or an invalid API key) — retrying a 429 storm just
    amplifies the outage.
    """
    if coder_agent.is_terminal_error(state.get("error") or ""):
        return "continue"

    validation = state.get("validation", {})
    status = validation.get("status", "validated")
    retry_count = state.get("retry_count", 0)

    if status == "rejected" and retry_count < 3:
        return "retry"
    return "continue"


# --- Interactive nodes (need a checkpointer; see build_graph) ---

def clarify_node(state: AgentState) -> AgentState:
    """Pause and ask the user when the planner flags real ambiguity.

    `interrupt()` suspends the run; the API resumes it with the user's
    answer (Command(resume=...)), which is appended to the question.
    """
    ask = (state.get("plan") or {}).get("clarifying_question")
    if ask:
        answer = interrupt({"type": "clarify", "question": ask})
        if answer:
            state["question"] = f"{state['question']}\nClarification from the user: {answer}"
            state["plan"]["clarifying_question"] = None
    return state


def confirm_cost_node(state: AgentState) -> AgentState:
    """Ask before running a query above the engine's cost threshold."""
    error = state.get("error") or ""
    if "CostConfirmationRequired" not in error:
        return state
    approved = interrupt({"type": "confirm_cost", "sql": state.get("sql_query"), "estimate": error})
    if approved:
        db, _ = _resources(state)
        result_df, err = db.execute_query(state["sql_query"], confirm_cost=True)
        state["result_df"], state["error"] = result_df, err
    else:
        state["error"] = "Cancelled: the query was not run because the cost was not approved."
    return state


def remember_node(state: AgentState) -> AgentState:
    """Record this turn so the next question in the conversation can refer to it."""
    result_df = state.get("result_df")
    preview = result_df.head(5).to_csv(index=False)[:600] if result_df is not None else None
    turn = {"question": state.get("display_question") or state["question"],
            "sql": state.get("sql_query"), "result_preview": preview}
    state["history"] = (state.get("history") or []) + [turn]
    return state


def _after_coder(state: AgentState) -> str:
    return "confirm_cost" if "CostConfirmationRequired" in (state.get("error") or "") else "visualizer"


# --- Build the Graph ---

def build_graph(interactive: bool = False) -> StateGraph:
    """Build the LangGraph agent workflow.

    `interactive=True` adds the human-in-the-loop nodes (clarifying
    question after planning, cost confirmation after coding). They use
    `interrupt()`, which requires compiling with a checkpointer (the API
    does); the Streamlit app and the benchmarks run the plain graph.
    """
    graph = StateGraph(AgentState)

    graph.add_node("schema", schema_node)
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("critic", critic_node)
    graph.add_node("visualizer", visualizer_node)
    graph.add_node("predictor", predictor_node)
    graph.add_node("storyteller", storyteller_node)
    graph.add_node("remember", remember_node)
    graph.add_node("decide_predict", lambda state: state)

    graph.set_entry_point("schema")
    graph.add_edge("schema", "planner")
    if interactive:
        graph.add_node("clarify", clarify_node)
        graph.add_node("confirm_cost", confirm_cost_node)
        graph.add_edge("planner", "clarify")
        graph.add_edge("clarify", "coder")
        graph.add_conditional_edges("coder", _after_coder,
                                    {"confirm_cost": "confirm_cost", "visualizer": "visualizer"})
        graph.add_edge("confirm_cost", "visualizer")
    else:
        graph.add_edge("planner", "coder")
        graph.add_edge("coder", "visualizer")
    graph.add_edge("visualizer", "critic")

    # Conditional: retry or continue after critic
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"retry": "coder", "continue": "decide_predict"},
    )
    graph.add_conditional_edges(
        "decide_predict",
        should_predict,
        {"predictor": "predictor", "storyteller": "storyteller"},
    )
    graph.add_edge("predictor", "storyteller")
    graph.add_edge("storyteller", "remember")
    graph.add_edge("remember", END)

    return graph


def turn_input(
    question: str,
    display_question: Optional[str] = None,
    skip_storyteller: bool = False,
) -> dict:
    """Per-turn fields, reset for every question. Conversation-level fields
    (schema, datasource_id, history) are left alone so a checkpointed
    thread keeps them across turns."""
    return {
        "question": question,
        "display_question": display_question or question,
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
        "skip_storyteller": skip_storyteller,
        "cache_hit": None,
    }


def _initial_state(
    question: str,
    schema: Optional[SemanticSchema],
    db: Database,
    df: pd.DataFrame,
    display_question: Optional[str],
    skip_storyteller: bool = False,
    datasource_id: Optional[str] = None,
) -> AgentState:
    """First-turn state. Registers (db, df) unless a datasource_id is given."""
    return {
        **turn_input(question, display_question, skip_storyteller),
        "schema": schema.model_dump() if schema is not None else None,
        "datasource_id": datasource_id or resources.register(db, df),
        "history": [],
    }


def run_analysis(
    question: str,
    schema: Optional[SemanticSchema],
    db: Database,
    df: pd.DataFrame,
    display_question: Optional[str] = None,
) -> AgentState:
    """
    Run the full agent pipeline for a question.

    `schema` may be None; the graph's schema node then profiles `df`.
    Returns the final state with all results.
    """
    app = build_graph().compile()
    initial_state = _initial_state(question, schema, db, df, display_question)

    run_id = tracing.new_run(question, dataset=db.table_name)
    try:
        result = app.invoke(initial_state)
        tracing.end_run(status="ok")
        if isinstance(result, dict):
            result["run_id"] = run_id
        return result
    except Exception as e:
        tracing.end_run(status="error", error=str(e))
        raise
    finally:
        resources.unregister(initial_state["datasource_id"])


def run_analysis_stream(
    question: str,
    schema: Optional[SemanticSchema],
    db: Database,
    df: pd.DataFrame,
    skip_storyteller: bool = False,
    display_question: Optional[str] = None,
):
    """
    Streaming version of run_analysis.

    Yields (node_name, merged_state) after each graph node executes, then
    a final ("done", final_state) event. Callers can use this to drive a
    live progress UI (e.g. st.status).
    """
    app = build_graph().compile()
    initial_state = _initial_state(
        question, schema, db, df, display_question, skip_storyteller,
    )

    run_id = tracing.new_run(question, dataset=db.table_name)
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
    finally:
        resources.unregister(initial_state["datasource_id"])

    yield "done", merged
