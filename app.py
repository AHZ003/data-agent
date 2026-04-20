"""DataAgent — Streamlit application entry point."""

import sys
import os
import json
import logging
from typing import Optional
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MAX_DISPLAY_ROWS, DEFAULT_TABLE_NAME, GOOGLE_API_KEY
from core.data_loader import load_file, parse_csv_string
from core.database import Database
from core.vision import extract_table_from_image
from agents.schema_agent import profile_dataframe
from agents.storyteller_agent import generate_full_report, report_to_markdown
from agents.orchestrator import run_analysis, run_analysis_stream
from models.analysis_plan import SemanticSchema, ColumnRole
from assets.ui import (
    inject_theme, topbar, section_label, hero, feature_card,
    result_header, narrative_block, chip, icon,
)
from assets.plotly_theme import register as register_plotly_theme
from assets.pdf_export import build_report_pdf
from assets.streaming import type_stream
from agents import storyteller_agent
from db import store
from core import tracing

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_ROWS_INGEST = 500_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dataagent")


# --- Page setup ---

st.set_page_config(
    page_title="DataAgent — AI Data Analyst",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_theme()
register_plotly_theme()
store.init_db()

# --- Query params ---
_qp = st.query_params
SHARED_ANALYSIS_ID = _qp.get("a")
EMBED_MODE = _qp.get("embed") in ("1", "true")
_QP_VIEW = _qp.get("view")
_QP_DATASET = _qp.get("ds")

if not GOOGLE_API_KEY:
    st.error(
        "Missing GOOGLE_API_KEY. Set it in your environment or .env file. "
        "Get a free key at https://aistudio.google.com/apikey."
    )
    st.stop()


# --- Session state ---

_DEFAULTS = {
    "db": None, "df": None, "schema": None,
    "messages": [], "analyses": [], "agent_log": [],
    "stream_target": None,
    "workspace_id": None,
    "dataset_id": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.workspace_id is None:
    _ws = store.ensure_default_workspace()
    st.session_state.workspace_id = _ws.id


# --- Helpers ---

def init_database(df: pd.DataFrame, table_name: str = DEFAULT_TABLE_NAME,
                  original_filename: Optional[str] = None, persist: bool = True):
    db = Database()
    db.load_dataframe(df, table_name)
    schema = profile_dataframe(df, table_name)
    st.session_state.db = db
    st.session_state.df = df
    st.session_state.schema = schema
    st.session_state.messages = []
    st.session_state.analyses = []
    st.session_state.agent_log = []
    st.session_state.stream_target = None

    st.query_params["view"] = "chat"
    st.query_params["ds"] = table_name

    if persist and st.session_state.workspace_id:
        try:
            dsid = store.save_dataset(
                workspace_id=st.session_state.workspace_id,
                name=table_name,
                df=df,
                schema_dict=schema.model_dump(),
                original_filename=original_filename,
            )
            st.session_state.dataset_id = dsid
        except Exception:
            log.exception("Failed to persist dataset")


def rehydrate_dataset(dataset_id: str) -> bool:
    data = store.load_dataset(dataset_id)
    if not data:
        return False
    df = data["df"]
    table_name = data["name"]
    db = Database()
    db.load_dataframe(df, table_name)
    schema = profile_dataframe(df, table_name)
    st.session_state.db = db
    st.session_state.df = df
    st.session_state.schema = schema
    st.session_state.dataset_id = dataset_id
    st.session_state.messages = []
    st.session_state.analyses = []
    st.session_state.agent_log = []
    st.session_state.stream_target = None
    return True


def clear_session():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v if not isinstance(v, list) else []
    for p in ("view", "ds"):
        st.query_params.pop(p, None)


def contextualize(question: str, messages: list) -> str:
    prior_qs = [m["content"] for m in messages if m["role"] == "user"][-3:]
    if not prior_qs:
        return question
    ctx = "\n".join(f"- {q}" for q in prior_qs)
    return (
        f"Earlier questions in this conversation:\n{ctx}\n\n"
        f"New question (may reference the above): {question}"
    )


_ROLE_ICON = {
    ColumnRole.MEASURE: "chart",
    ColumnRole.DIMENSION: "layers",
    ColumnRole.TIME_AXIS: "bolt",
    ColumnRole.IDENTIFIER: "search",
    ColumnRole.TEXT: "pen",
}


def render_profile(schema: SemanticSchema):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{schema.row_count:,}")
    c2.metric("Columns", schema.column_count)
    c3.metric("Measures", sum(1 for c in schema.columns if c.role == ColumnRole.MEASURE))
    c4.metric("Dimensions", sum(1 for c in schema.columns if c.role == ColumnRole.DIMENSION))

    with st.expander("Schema details", expanded=False):
        for col in schema.columns:
            ic = _ROLE_ICON.get(col.role, "file")
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:0.75rem;padding:0.6rem 0;border-bottom:1px solid var(--border);">
                    <span style="color:var(--accent);">{icon(ic, 15)}</span>
                    <div style="flex:1;">
                        <div style="font-weight:500;color:var(--text);font-size:0.875rem;">{col.name}</div>
                        <div style="font-size:0.72rem;color:var(--text-mute);">
                            {col.dtype} · {col.role.value} · {col.null_percentage}% null
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_trace_panel(run_id: str):
    try:
        spans = tracing.read_spans(run_id=run_id)
        summary = tracing.run_summary(run_id)
    except Exception as e:
        st.caption(f"Trace unavailable: {e}")
        return

    agent_spans = [s for s in spans if "span_id" in s]
    if not agent_spans:
        st.caption("No trace spans recorded for this run.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{summary['total_duration_ms'] / 1000:.2f}s")
    col2.metric("Tokens in", f"{summary['total_tokens_in']:,}")
    col3.metric("Tokens out", f"{summary['total_tokens_out']:,}")
    col4.metric("Est. cost", f"${summary['total_cost_usd']:.5f}")

    rows = []
    for s in agent_spans:
        rows.append({
            "agent": s.get("agent_name", ""),
            "duration_ms": s.get("duration_ms", 0),
            "tokens_in": s.get("tokens_in", 0),
            "tokens_out": s.get("tokens_out", 0),
            "cost_usd": s.get("cost_usd", 0.0),
            "status": s.get("status", ""),
            "model": s.get("model") or "",
        })
    import pandas as _pd
    df_trace = _pd.DataFrame(rows)
    st.dataframe(df_trace, use_container_width=True, hide_index=True)
    st.caption(f"run_id: `{run_id}` — trace file: `outputs/traces/`")


def _session_cost_summary() -> dict:
    total_ms = 0.0
    total_in = 0
    total_out = 0
    total_cost = 0.0
    n_runs = 0
    for a in st.session_state.get("analyses", []):
        rid = a.get("run_id")
        if not rid:
            continue
        try:
            s = tracing.run_summary(rid)
        except Exception:
            continue
        if s.get("n_spans", 0) == 0:
            continue
        n_runs += 1
        total_ms += s.get("total_duration_ms", 0)
        total_in += s.get("total_tokens_in", 0)
        total_out += s.get("total_tokens_out", 0)
        total_cost += s.get("total_cost_usd", 0.0)
    return {
        "n_runs": n_runs,
        "total_duration_ms": round(total_ms, 2),
        "total_tokens_in": total_in,
        "total_tokens_out": total_out,
        "total_cost_usd": round(total_cost, 6),
    }


def render_assistant_body(analysis: dict, should_stream: bool):
    """Render chart + narrative + chips + expanders inside an assistant message."""
    validation = analysis.get("validation") or {}
    conf = validation.get("confidence")

    # Narrative
    if analysis.get("narrative"):
        if should_stream:
            st.write_stream(type_stream(analysis["narrative"], words_per_chunk=2, delay=0.025))
        else:
            st.markdown(analysis["narrative"])

    # Chart
    if analysis.get("chart") is not None:
        st.plotly_chart(analysis["chart"], use_container_width=True, config={"displayModeBar": False})

    if analysis.get("prediction_chart") is not None:
        st.plotly_chart(analysis["prediction_chart"], use_container_width=True, config={"displayModeBar": False})

    if analysis.get("prediction") and "explanation" in (analysis.get("prediction") or {}):
        st.info(analysis["prediction"]["explanation"])

    # Warnings
    if validation.get("warnings"):
        for w in validation["warnings"]:
            st.warning(w)

    # Metadata bar
    meta_cols = st.columns([1, 1, 1, 2])
    if conf is not None:
        kind = "ok" if conf >= 70 else ("warn" if conf >= 40 else "err")
        colors = {"ok": "#00B894", "warn": "#E17055", "err": "#D63031"}
        meta_cols[0].markdown(
            f'<div style="font-size:0.75rem;color:var(--text-mute);">Confidence</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:{colors[kind]};">{conf}%</div>',
            unsafe_allow_html=True,
        )
    if analysis.get("result_df") is not None:
        n_rows = len(analysis["result_df"])
        meta_cols[1].markdown(
            f'<div style="font-size:0.75rem;color:var(--text-mute);">Rows</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:var(--text);">{n_rows:,}</div>',
            unsafe_allow_html=True,
        )
    if analysis.get("sql_query"):
        sql_label = "pandas" if analysis["sql_query"].startswith("--") else "SQL"
        meta_cols[2].markdown(
            f'<div style="font-size:0.75rem;color:var(--text-mute);">Method</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:var(--text);">{sql_label}</div>',
            unsafe_allow_html=True,
        )

    # Expandable details
    with st.expander("View details", expanded=False):
        tab_names = []
        if analysis.get("result_df") is not None and len(analysis["result_df"]) > 0:
            tab_names.append("Data")
        if analysis.get("sql_query"):
            tab_names.append("Query")
        if analysis.get("run_id"):
            tab_names.append("Trace")
        if tab_names:
            tabs = st.tabs(tab_names)
            tab_idx = 0
            if "Data" in tab_names:
                with tabs[tab_idx]:
                    st.dataframe(analysis["result_df"], use_container_width=True, height=300)
                tab_idx += 1
            if "Query" in tab_names:
                with tabs[tab_idx]:
                    st.code(analysis["sql_query"], language="sql")
                tab_idx += 1
            if "Trace" in tab_names:
                with tabs[tab_idx]:
                    _render_trace_panel(analysis["run_id"])


# --- Shared analysis view ---

if SHARED_ANALYSIS_ID:
    shared = store.load_analysis(SHARED_ANALYSIS_ID)
    if not shared:
        st.error("Analysis not found. It may have been deleted.")
        st.stop()

    ds = store.load_dataset(shared["dataset_id"])
    ds_name = ds["name"].replace("_", " ").title() if ds else "Analysis"

    st.markdown(
        f"""
        <div class="da-workspace-head">
            <div class="da-hero-eyebrow">{icon("bolt", 12)} Shared analysis</div>
            <h1 class="da-workspace-title">{ds_name}</h1>
            <p class="da-workspace-sub">{shared['question']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    result_header(shared["question"])
    render_assistant_body(shared, should_stream=False)
    st.stop()


# --- File/image ingestion helpers ---

def _handle_file_upload(uploaded_file):
    """Process an uploaded file and initialize the database."""
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error(
            f"File too large ({uploaded_file.size / 1024 / 1024:.1f} MB). "
            f"Limit is {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
        )
        return False
    df = load_file(uploaded_file)
    if len(df) > MAX_ROWS_INGEST:
        st.warning(f"Dataset truncated to {MAX_ROWS_INGEST:,} rows (uploaded {len(df):,}).")
        df = df.head(MAX_ROWS_INGEST)
    table_name = uploaded_file.name.rsplit(".", 1)[0].lower().replace(" ", "_")
    with st.spinner("Profiling dataset..."):
        init_database(df, table_name, original_filename=uploaded_file.name)
    log.info("Loaded %s rows=%d cols=%d", table_name, len(df), len(df.columns))
    return True


# --- Navigation: handle browser back button ---
if st.session_state.schema is not None and _QP_VIEW != "chat":
    clear_session()
    st.rerun()


# =====================================================================
# LANDING PAGE
# =====================================================================

if st.session_state.schema is None:
    hero(
        title="Ask your data anything.",
        subtitle="Upload a spreadsheet and DataAgent's multi-agent system will profile, "
                 "query, visualize, predict, and narrate — grounded by a statistical critic.",
        eyebrow="AI Data Analyst",
    )

    # Feature cards
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        feature_card("database", "Schema profiling",
                     "Infers column roles, distributions, and date ranges on upload.")
    with f2:
        feature_card("brain", "Plan, code, critique",
                     "Planner decomposes, Coder writes SQL, Critic validates rigor.")
    with f3:
        feature_card("chart", "Visualize & narrate",
                     "Picks the right chart and explains findings in plain English.")

    f4, f5, f6 = st.columns(3, gap="medium")
    with f4:
        feature_card("predict", "Forecast & cluster",
                     "Time-series forecasting, segmentation, and anomaly detection.")
    with f5:
        feature_card("shield", "Grounded by RAG",
                     "Statistical knowledge base prevents hallucinated methods.")
    with f6:
        feature_card("document", "Exec-ready reports",
                     "One click turns your session into a branded PDF deliverable.")

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # --- Quick start: sample datasets ---
    section_label("Quick start — load a sample dataset")
    qs1, qs2, qs3 = st.columns(3, gap="medium")
    _samples = {
        "E-Commerce Transactions": "data/ecommerce_transactions.csv",
        "Superstore Sales": "data/superstore_sales.csv",
        "Employee Data": "data/employee_data.csv",
    }
    for col, (name, path) in zip([qs1, qs2, qs3], _samples.items()):
        if col.button(name, key=f"qs_{name}", use_container_width=True, type="primary"):
            if os.path.exists(path):
                with st.spinner("Loading & profiling..."):
                    _df = pd.read_csv(path)
                    init_database(_df, name.lower().replace(" ", "_"),
                                  original_filename=os.path.basename(path))
                st.rerun()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # --- Upload your own ---
    section_label("Or upload your own file")
    landing_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        key="landing_upload",
    )
    if landing_file is not None:
        try:
            if _handle_file_upload(landing_file):
                st.rerun()
        except Exception as e:
            st.error(f"Couldn't read that file — {e}")

    # --- Or extract from image ---
    with st.expander("Extract table from image"):
        landing_image = st.file_uploader(
            "Upload a photo of a table",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="landing_image",
            help="Upload a photo of a table — Gemini will OCR it",
        )
        if landing_image is not None:
            st.image(landing_image, use_container_width=True)
            with st.spinner("Extracting table from image..."):
                image_bytes = landing_image.read()
                mime = f"image/{landing_image.name.rsplit('.', 1)[-1].lower()}"
                if mime == "image/jpg":
                    mime = "image/jpeg"
                csv_string, error = extract_table_from_image(image_bytes, mime)
            if error:
                st.error(error)
            elif csv_string:
                df = parse_csv_string(csv_string)
                if df is not None:
                    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
                    if st.button("Use this data", type="primary"):
                        init_database(edited_df, "image_data", original_filename="image_extract")
                        st.rerun()

    # --- Previously loaded datasets ---
    _library = store.list_datasets(st.session_state.workspace_id)
    if _library:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        section_label("Recent datasets")
        cols = st.columns(min(len(_library), 3), gap="medium")
        for i, ds in enumerate(_library[:6]):
            with cols[i % 3]:
                if st.button(
                    f"{ds.name.replace('_', ' ').title()}",
                    key=f"lib_{ds.id}",
                    use_container_width=True,
                ):
                    with st.spinner("Loading dataset..."):
                        rehydrate_dataset(ds.id)
                    st.rerun()


# =====================================================================
# ANALYSIS / CHAT PAGE
# =====================================================================

else:
    schema = st.session_state.schema

    nav_l, nav_c, nav_r = st.columns([1, 4, 1])
    if nav_l.button("Back", key="nav_back", use_container_width=True):
        clear_session()
        st.rerun()
    nav_c.markdown(
        f'<div style="text-align:center;padding-top:0.3rem;">'
        f'<span style="font-weight:600;color:var(--text);font-size:1rem;">'
        f'{schema.table_name.replace("_", " ").title()}</span>'
        f'<span style="color:var(--text-mute);font-size:0.82rem;margin-left:0.75rem;">'
        f'{schema.row_count:,} rows · {schema.column_count} columns</span></div>',
        unsafe_allow_html=True,
    )
    if nav_r.button("New chat", key="nav_new", use_container_width=True):
        st.session_state.messages = []
        st.session_state.analyses = []
        st.session_state.agent_log = []
        st.rerun()

    # --- Dataset overview (collapsed) ---
    with st.expander("Dataset overview", expanded=False):
        render_profile(schema)
        st.dataframe(st.session_state.df.head(MAX_DISPLAY_ROWS), use_container_width=True)

    # --- Session info bar (replaces sidebar cost/activity) ---
    _cost = _session_cost_summary()
    if _cost["n_runs"] > 0:
        cost_str = f"${_cost['total_cost_usd']:.5f}" if _cost["total_cost_usd"] > 0 else "—"
        st.markdown(
            f"""
            <div class="da-info-bar">
                <div class="da-info-item">{icon("bolt", 13)} <strong>{_cost['n_runs']}</strong> runs</div>
                <div class="da-info-item">Tokens: <strong>{_cost['total_tokens_in'] + _cost['total_tokens_out']:,}</strong></div>
                <div class="da-info-item">Cost: <strong>{cost_str}</strong></div>
                <div class="da-info-item">Duration: <strong>{_cost['total_duration_ms']/1000:.1f}s</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Starter suggestions ---
    if not st.session_state.messages and schema.suggested_questions:
        section_label("Starter questions")
        st.markdown('<div class="da-suggest-anchor"></div>', unsafe_allow_html=True)
        cols = st.columns(3, gap="medium")
        for i, q in enumerate(schema.suggested_questions[:6]):
            if cols[i % 3].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["_pending_q"] = q
                st.rerun()

    # --- Chat history ---
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                if msg.get("error"):
                    st.error(msg["error"])
                elif msg.get("analysis"):
                    should_stream = (st.session_state.stream_target == idx)
                    render_assistant_body(msg["analysis"], should_stream=should_stream)

    if st.session_state.stream_target is not None:
        st.session_state.stream_target = None

    # --- Chat input ---
    pending = st.session_state.pop("_pending_q", None)
    prompt = st.chat_input("Ask a question about your data...")
    if pending and not prompt:
        prompt = pending

    # --- Action row: report export ---
    with st.container():
        col_a, col_b, col_c = st.columns([1, 1, 2])
        report_clicked = col_a.button("Export PDF report", use_container_width=True,
                                       disabled=not st.session_state.analyses)
        md_clicked = col_b.button("Export Markdown", use_container_width=True,
                                   disabled=not st.session_state.analyses)

    # --- Handle new chat turn ---
    if prompt:
        log.info("User: %s", prompt[:80])
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            contextual_q = contextualize(prompt, st.session_state.messages)

            phase_labels = {
                "planner":       ("Planning analysis",        "Decomposing the question into steps"),
                "coder":          ("Writing SQL",              "Generating and executing the query"),
                "visualizer":     ("Choosing visualization",   "Selecting the right chart type"),
                "critic":         ("Validating rigor",         "Statistical and sanity checks"),
                "decide_predict": ("Routing",                  "Deciding if ML prediction is needed"),
                "predictor":      ("Running ML model",         "Forecast, cluster, or anomaly detection"),
                "storyteller":    ("Writing narrative",        "Translating results into business English"),
            }

            result = None
            with st.status("Agents working...", expanded=True) as status:
                for node, state in run_analysis_stream(
                    question=contextual_q,
                    schema=schema,
                    db=st.session_state.db,
                    df=st.session_state.df,
                    skip_storyteller=True,
                ):
                    if node == "done":
                        result = state
                        status.update(label="Writing narrative", state="running")
                        break
                    label, detail = phase_labels.get(node, (node.title(), ""))
                    status.update(label=label)
                    st.write(f"**{label}** — {detail}")

            if result is None:
                raise RuntimeError("Agent pipeline returned no result")

            result_df = result.get("result_df")
            validation = result.get("validation") or {}
            val_status = validation.get("status", "validated")
            has_data = result_df is not None and len(result_df) > 0

            with st.chat_message("assistant"):
                if not has_data or val_status == "rejected":
                    error_msg = result.get("error") or ""
                    if "QuotaExhausted" in error_msg:
                        narrative = "The AI service quota has been exceeded. Please try again later."
                    elif not has_data:
                        narrative = (
                            "The query returned no results. This may mean the "
                            "filters are too restrictive, or the question doesn't "
                            "match the available data. Try rephrasing."
                        )
                    else:
                        narrative = (
                            "The analysis was flagged as unreliable by the "
                            "validation agent (confidence too low). The results "
                            "may be inaccurate — consider rephrasing the question."
                        )
                    st.warning(narrative)
                else:
                    result_summary = result_df.head(20).to_string()
                    narrative = st.write_stream(
                        storyteller_agent.stream_narrative(
                            question=contextual_q,
                            sql_query=result.get("sql_query", "") or "",
                            result_summary=result_summary,
                            chart_description=(result.get("chart_config") or {}).get("chart_type", ""),
                            validation_warnings=validation.get("warnings", []),
                            prediction_info=result.get("prediction"),
                        )
                    )

                # Show chart inline on first render
                if result.get("chart") is not None:
                    st.plotly_chart(result["chart"], use_container_width=True, config={"displayModeBar": False})
                if result.get("prediction_chart") is not None:
                    st.plotly_chart(result["prediction_chart"], use_container_width=True, config={"displayModeBar": False})

            analysis = {
                "question": prompt,
                "sql_query": result.get("sql_query"),
                "result_df": result.get("result_df"),
                "chart": result.get("chart"),
                "prediction_chart": result.get("prediction_chart"),
                "prediction": result.get("prediction"),
                "narrative": narrative or "",
                "validation": result.get("validation"),
                "error": result.get("error"),
                "run_id": result.get("run_id"),
            }
            analysis_id = None
            if st.session_state.dataset_id:
                try:
                    analysis_id = store.save_analysis(
                        workspace_id=st.session_state.workspace_id,
                        dataset_id=st.session_state.dataset_id,
                        analysis={**analysis, "question": prompt},
                    )
                except Exception:
                    log.exception("Failed to persist analysis")

            analysis["id"] = analysis_id
            st.session_state.analyses.append(analysis)
            st.session_state.agent_log.extend(result.get("agent_log", []))
            st.session_state.messages.append({
                "role": "assistant",
                "question": prompt,
                "analysis": analysis,
            })
            st.session_state.stream_target = None
        except Exception as e:
            log.exception("Analysis failed")
            err_str = str(e)
            if any(p in err_str for p in ("429", "RESOURCE_EXHAUSTED", "quota", "rate limit")):
                user_error = (
                    "The AI service quota has been exceeded. The free tier allows "
                    "a limited number of requests per minute/day. Please wait a "
                    "moment and try again, or upgrade your Gemini API plan."
                )
            else:
                user_error = f"The agent crew hit an error. Try rephrasing your question."
            log.error("User-facing error: %s | Raw: %s", user_error, err_str[:300])
            st.session_state.messages.append({
                "role": "assistant",
                "question": prompt,
                "error": user_error,
            })
        st.rerun()

    # --- Report export handlers ---
    if (report_clicked or md_clicked) and st.session_state.analyses:
        with st.spinner("Assembling report..."):
            try:
                schema_summary = json.dumps(schema.model_dump(), indent=2, default=str)
                session_data = {
                    "schema_summary": schema_summary[:2000],
                    "analyses": [
                        {
                            "question": a["question"],
                            "sql": a.get("sql_query", ""),
                            "result_summary": (
                                a["result_df"].head(10).to_string()
                                if a.get("result_df") is not None else ""
                            ),
                            "narrative": a.get("narrative", ""),
                            "warnings": (a.get("validation") or {}).get("warnings", []),
                        }
                        for a in st.session_state.analyses
                    ],
                    "predictions": [
                        a["prediction"] for a in st.session_state.analyses if a.get("prediction")
                    ],
                }
                report = generate_full_report(session_data)

                if report_clicked:
                    pdf_bytes = build_report_pdf(
                        report,
                        dataset_name=schema.table_name.replace("_", " ").title(),
                    )
                    st.success("PDF report ready.")
                    st.download_button(
                        "Download PDF",
                        data=pdf_bytes,
                        file_name=f"dataagent_report_{schema.table_name}.pdf",
                        mime="application/pdf",
                        type="primary",
                    )
                else:
                    md = report_to_markdown(report)
                    st.success("Markdown report ready.")
                    st.download_button(
                        "Download Markdown",
                        data=md,
                        file_name=f"dataagent_report_{schema.table_name}.md",
                        mime="text/markdown",
                    )
            except Exception as e:
                log.exception("Report generation failed")
                st.error(f"Report generation failed: {e}")
