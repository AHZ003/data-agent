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
    inject_theme, brand, sidebar_label, hero, feature_card,
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
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
register_plotly_theme()
store.init_db()

# --- Query params (share + embed) ---
_qp = st.query_params
SHARED_ANALYSIS_ID = _qp.get("a")
EMBED_MODE = _qp.get("embed") in ("1", "true")

if EMBED_MODE:
    st.markdown(
        "<style>section[data-testid='stSidebar'], [data-testid='collapsedControl']{display:none!important;}"
        ".block-container{max-width:100%!important;padding:1rem 1.5rem!important;}</style>",
        unsafe_allow_html=True,
    )

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

# Ensure a default workspace exists and is selected
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
    """Load a persisted dataset back into session state."""
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


def contextualize(question: str, messages: list) -> str:
    """Prepend recent conversation context so the agent crew handles follow-ups."""
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
    """Inline expander showing the structured-trace spans for one run.

    Uses `core.tracing.read_spans(run_id=...)` — the same JSONL file the
    eval harness writes to — so the user can see exactly which agent
    spent how much time, how many tokens, and how many $ this question
    just cost. This is the minimum-viable observability UX that turns
    "did the agent work?" into "what did the agent do?".
    """
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
    """Sum duration + cost across every analysis in this session that
    has a run_id. Used by the sidebar `Session cost` card.
    """
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

    # --- Narrative ---
    if analysis.get("narrative"):
        if should_stream:
            st.write_stream(type_stream(analysis["narrative"], words_per_chunk=2, delay=0.025))
        else:
            st.markdown(analysis["narrative"])

    # --- Chart ---
    if analysis.get("chart") is not None:
        st.plotly_chart(analysis["chart"], use_container_width=True, config={"displayModeBar": False})

    if analysis.get("prediction_chart") is not None:
        st.plotly_chart(analysis["prediction_chart"], use_container_width=True, config={"displayModeBar": False})

    if analysis.get("prediction") and "explanation" in (analysis.get("prediction") or {}):
        st.info(analysis["prediction"]["explanation"])

    # --- Warnings ---
    if validation.get("warnings"):
        for w in validation["warnings"]:
            st.warning(w)

    # --- Metadata bar ---
    meta_cols = st.columns([1, 1, 1, 2])
    if conf is not None:
        kind = "ok" if conf >= 70 else ("warn" if conf >= 40 else "err")
        colors = {"ok": "#34D399", "warn": "#F59E0B", "err": "#F87171"}
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

    # --- Expandable details ---
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


# --- Shared analysis view (single-page render via ?a=<id>) ---

if SHARED_ANALYSIS_ID:
    shared = store.load_analysis(SHARED_ANALYSIS_ID)
    if not shared:
        st.error("Analysis not found. It may have been deleted.")
        st.stop()

    ds = store.load_dataset(shared["dataset_id"])
    ds_name = ds["name"].replace("_", " ").title() if ds else "Analysis"

    if not EMBED_MODE:
        brand()
    st.markdown(
        f"""
        <div class="da-workspace-head">
            <div class="da-hero-eyebrow">{icon("bolt", 12, "#B8A5FF")} Shared analysis</div>
            <h1 class="da-workspace-title">{ds_name}</h1>
            <p class="da-workspace-sub">{shared['question']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    result_header(shared["question"])
    render_assistant_body(shared, should_stream=False)
    st.stop()


# --- Sidebar ---

with st.sidebar:
    brand()

    # --- Workspaces ---
    sidebar_label("Workspace")
    _workspaces = store.list_workspaces()
    _ws_names = [w.name for w in _workspaces]
    _current_idx = next(
        (i for i, w in enumerate(_workspaces) if w.id == st.session_state.workspace_id),
        0,
    )
    _selected = st.selectbox(
        "workspace",
        _ws_names,
        index=_current_idx if _ws_names else 0,
        label_visibility="collapsed",
        key="ws_select",
    )
    if _selected and _workspaces:
        _new_id = next(w.id for w in _workspaces if w.name == _selected)
        if _new_id != st.session_state.workspace_id:
            st.session_state.workspace_id = _new_id
            clear_session()
            st.session_state.workspace_id = _new_id
            st.rerun()

    with st.popover("＋  New workspace", use_container_width=True):
        _new_name = st.text_input("Name", key="new_ws_name", label_visibility="collapsed",
                                   placeholder="Q2 Revenue Review")
        if st.button("Create", key="create_ws", use_container_width=True, type="primary"):
            if _new_name.strip():
                ws = store.create_workspace(_new_name.strip())
                st.session_state.workspace_id = ws.id
                clear_session()
                st.session_state.workspace_id = ws.id
                st.rerun()

    sidebar_label("Data source")
    uploaded_file = st.file_uploader(
        "Upload spreadsheet",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    uploaded_image = st.file_uploader(
        "Extract from image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="Upload a photo of a table — Gemini will OCR it",
    )

    sidebar_label("Sample datasets")
    samples = {
        "Superstore Sales": "data/superstore_sales.csv",
        "E-Commerce Transactions": "data/ecommerce_transactions.csv",
        "Employee Data": "data/employee_data.csv",
    }
    sample_choice = st.selectbox(
        "Sample",
        ["Select…"] + list(samples.keys()),
        label_visibility="collapsed",
    )
    if sample_choice != "Select…":
        sample_path = samples[sample_choice]
        if os.path.exists(sample_path):
            if st.button("Load dataset", use_container_width=True, type="primary"):
                with st.spinner("Loading & profiling…"):
                    df = pd.read_csv(sample_path)
                    init_database(df, sample_choice.lower().replace(" ", "_"),
                                  original_filename=os.path.basename(sample_path))
                st.rerun()
        else:
            st.caption(f"Missing: {sample_path}")

    # --- Dataset library ---
    _library = store.list_datasets(st.session_state.workspace_id)
    if _library:
        sidebar_label("Dataset library")
        for ds in _library[:8]:
            is_active = ds.id == st.session_state.dataset_id
            label = f"{'●' if is_active else '○'}  {ds.name[:22]}"
            if st.button(label, key=f"ds_{ds.id}", use_container_width=True):
                with st.spinner("Loading dataset…"):
                    rehydrate_dataset(ds.id)
                st.rerun()

    if st.session_state.df is not None:
        sidebar_label("Active dataset")
        st.markdown(
            f"""
            <div class="da-card">
                <div class="da-card-title">{st.session_state.schema.table_name}</div>
                <div class="da-card-sub">{st.session_state.schema.row_count:,} rows · {st.session_state.schema.column_count} cols</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Clear session", use_container_width=True):
            clear_session()
            st.rerun()

    if st.session_state.agent_log:
        sidebar_label("Agent activity")
        for entry in reversed(st.session_state.agent_log[-6:]):
            ok = entry.get("status") == "success"
            kind = "ok" if ok else "err"
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;align-items:center;padding:0.4rem 0;border-bottom:1px solid var(--border);font-size:0.78rem;">
                    <span style="color:var(--text-dim);">{entry['agent_name']}</span>
                    {chip(f"{entry['duration_seconds']:.1f}s", kind)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Session cost card — sums across all runs in this chat.
    _cost = _session_cost_summary()
    if _cost["n_runs"] > 0:
        sidebar_label("Session cost")
        st.markdown(
            f"""
            <div style="padding:0.6rem 0.75rem;border:1px solid var(--border);border-radius:10px;font-size:0.78rem;">
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:var(--text-dim);">Runs</span>
                    <span>{_cost['n_runs']}</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:var(--text-dim);">Tokens in</span>
                    <span>{_cost['total_tokens_in']:,}</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="color:var(--text-dim);">Tokens out</span>
                    <span>{_cost['total_tokens_out']:,}</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:0.3rem;padding-top:0.3rem;border-top:1px solid var(--border);">
                    <span style="color:var(--text-dim);">Est. cost</span>
                    <span><strong>${_cost['total_cost_usd']:.5f}</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- File/image ingestion ---

if uploaded_file is not None and st.session_state.df is None:
    try:
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            st.error(
                f"File too large ({uploaded_file.size / 1024 / 1024:.1f} MB). "
                f"Limit is {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
            )
        else:
            df = load_file(uploaded_file)
            if len(df) > MAX_ROWS_INGEST:
                st.warning(
                    f"Dataset truncated to {MAX_ROWS_INGEST:,} rows "
                    f"(uploaded {len(df):,})."
                )
                df = df.head(MAX_ROWS_INGEST)
            table_name = uploaded_file.name.rsplit(".", 1)[0].lower().replace(" ", "_")
            with st.spinner("Profiling dataset…"):
                init_database(df, table_name, original_filename=uploaded_file.name)
            log.info("Loaded %s rows=%d cols=%d", table_name, len(df), len(df.columns))
            st.rerun()
    except Exception as e:
        log.exception("Upload failed")
        st.error(f"Couldn't read that file — {e}")

if uploaded_image is not None and st.session_state.df is None:
    st.image(uploaded_image, use_container_width=True)
    with st.spinner("Extracting table from image…"):
        image_bytes = uploaded_image.read()
        mime = f"image/{uploaded_image.name.rsplit('.', 1)[-1].lower()}"
        if mime == "image/jpg":
            mime = "image/jpeg"
        csv_string, error = extract_table_from_image(image_bytes, mime)
    if error:
        st.error(error)
    elif csv_string:
        df = parse_csv_string(csv_string)
        if df is not None:
            st.subheader("Extracted table")
            edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
            if st.button("Use this data", type="primary"):
                init_database(edited_df, "image_data", original_filename="image_extract")
                st.rerun()


# --- Main content ---

if st.session_state.schema is None:
    # Landing
    hero(
        title="Ask your data anything.",
        subtitle="Upload a spreadsheet and DataAgent's multi-agent system profiles, queries, "
                 "visualizes, predicts, and narrates findings — grounded by a statistical critic.",
        eyebrow="AI Data Analyst",
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        feature_card("database", "Schema profiling",
                     "Infers column roles, distributions, and date ranges on upload.")
    with f2:
        feature_card("brain", "Plan → code → critique",
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

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Quick-start buttons on the landing page so users don't need the sidebar
    st.markdown(
        '<div class="da-sidebar-label" style="margin:0 0 0.5rem 0;">Quick start — load a sample dataset</div>',
        unsafe_allow_html=True,
    )
    qs1, qs2, qs3 = st.columns(3, gap="medium")
    _samples_landing = {
        "E-Commerce Transactions": "data/ecommerce_transactions.csv",
        "Superstore Sales": "data/superstore_sales.csv",
        "Employee Data": "data/employee_data.csv",
    }
    for col, (name, path) in zip([qs1, qs2, qs3], _samples_landing.items()):
        if col.button(name, key=f"qs_{name}", use_container_width=True, type="primary"):
            if os.path.exists(path):
                with st.spinner("Loading & profiling…"):
                    _df = pd.read_csv(path)
                    init_database(_df, name.lower().replace(" ", "_"),
                                  original_filename=os.path.basename(path))
                st.rerun()

    st.markdown(
        '<div style="color:var(--text-mute);font-size:0.82rem;margin-top:1rem;">'
        'Or upload your own file from the sidebar.'
        '</div>',
        unsafe_allow_html=True,
    )

else:
    schema = st.session_state.schema

    # Compact workspace header
    st.markdown(
        f"""
        <div class="da-workspace-head">
            <h1 class="da-workspace-title">{schema.table_name.replace("_", " ").title()}</h1>
            <p class="da-workspace-sub">{schema.row_count:,} rows · {schema.column_count} columns</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Dataset overview", expanded=False):
        render_profile(schema)
        st.dataframe(st.session_state.df.head(MAX_DISPLAY_ROWS), use_container_width=True)

    # Starter suggestions (only when the chat is empty)
    if not st.session_state.messages and schema.suggested_questions:
        st.markdown(
            '<div class="da-sidebar-label" style="margin:1rem 0 0.5rem 0;">Starter questions</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="da-suggest-anchor"></div>', unsafe_allow_html=True)
        cols = st.columns(3, gap="medium")
        for i, q in enumerate(schema.suggested_questions[:6]):
            if cols[i % 3].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["_pending_q"] = q
                st.rerun()

    # Render chat history
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

    # Clear one-shot stream flag after render
    if st.session_state.stream_target is not None:
        st.session_state.stream_target = None

    # Chat input
    pending = st.session_state.pop("_pending_q", None)
    prompt = st.chat_input("Ask a follow-up question…")
    if pending and not prompt:
        prompt = pending

    # Action row: report export
    with st.container():
        col_a, col_b, col_c = st.columns([1, 1, 2])
        report_clicked = col_a.button("Export PDF report", use_container_width=True,
                                       disabled=not st.session_state.analyses)
        md_clicked = col_b.button("Export Markdown", use_container_width=True,
                                   disabled=not st.session_state.analyses)

    # Handle new chat turn
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
            with st.status("Agents working…", expanded=True) as status:
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
                    # Don't stream a narrative when there's no data or the
                    # critic rejected — the LLM would hallucinate from
                    # the question alone, producing a confident-sounding
                    # answer backed by nothing.
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
            # Narrative was already streamed live via stream_narrative above,
            # so leave stream_target=None to avoid a fake-stream replay on rerun.
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
        with st.spinner("Assembling report…"):
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
