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
    inject_theme, section_label, hero, feature_card,
    result_header, narrative_block, chip, icon,
)
from assets.plotly_theme import register as register_plotly_theme
from assets.pdf_export import build_report_pdf
from assets.streaming import type_stream
from agents import storyteller_agent
from db import store
from core import pii, tracing
from core.demo import SAMPLE_QUESTIONS, DemoLimits, VisitorBudget, load_sample_workspace
from core.llm import api_key_override
import feedback

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_ROWS_INGEST = 500_000
DEMO = DemoLimits.from_env()
# The store has one workspace for everyone. On the public demo that would let
# any visitor open another visitor's uploads ("Recent datasets"), so nothing
# is persisted there.
PERSIST = not DEMO.enabled


@st.cache_resource
def _visitor_budget() -> VisitorBudget:
    return VisitorBudget()


def _visitor_key() -> Optional[str]:
    """A reload starts a new session, so the demo question cap is counted per
    visitor (client IP + user agent) as well as per session."""
    try:
        h = st.context.headers
        ip = (h.get("X-Forwarded-For") or "").split(",")[0].strip() or (st.context.ip_address or "")
        return VisitorBudget.visitor_key(ip, h.get("User-Agent", "")) if ip else None
    except Exception:
        return None
_SAMPLES = {
    "E-Commerce Transactions": "data/ecommerce_transactions.csv",
    "Superstore Sales": "data/superstore_sales.csv",
    "Employee Data": "data/employee_data.csv",
}


def md_safe(text: str) -> str:
    """Streamlit renders $...$ as LaTeX: "$77,623 ... $38.90" became one
    garbled italic formula. Escape dollar signs for display only."""
    return (text or "").replace("$", "\\$")


def _escaped(chunks, sink: list):
    """Stream chunks escaped for display while keeping the raw text in `sink`."""
    for c in chunks:
        sink.append(c)
        yield md_safe(c)

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
    "user_api_key": "",
    "questions_asked": 0,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.workspace_id is None:
    _ws = store.ensure_default_workspace()
    st.session_state.workspace_id = _ws.id


# --- Helpers ---

def init_database(df: pd.DataFrame, table_name: str = DEFAULT_TABLE_NAME,
                  original_filename: Optional[str] = None, persist: bool = True,
                  extra_tables: Optional[dict] = None):
    """Load the primary table (and any extra uploaded tables) and profile it."""
    db = Database()
    db.load_dataframe(df, table_name)
    for name, extra in (extra_tables or {}).items():
        db.load_dataframe(extra, name)
    schema = profile_dataframe(df, table_name)
    _set_workspace(db, df, schema, table_name)
    if persist and PERSIST and st.session_state.workspace_id:
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


def init_sample_workspace():
    """Chinook: a real 11-table music-store database, opened read-only."""
    from agents.orchestrator import _sample_primary_table
    db = load_sample_workspace()
    df = _sample_primary_table(db)
    schema = profile_dataframe(df, db.table_name, suggest_questions=False)
    schema.suggested_questions = list(SAMPLE_QUESTIONS)
    _set_workspace(db, df, schema, "chinook")
    st.session_state.dataset_id = None


def _set_workspace(db, df, schema, name: str):
    st.session_state.db = db
    st.session_state.df = df
    st.session_state.schema = schema
    st.session_state.messages = []
    st.session_state.analyses = []
    st.session_state.agent_log = []
    st.session_state.stream_target = None
    st.query_params["view"] = "chat"
    st.query_params["ds"] = name


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
    keep = {"user_api_key", "questions_asked", "workspace_id"}
    for k, v in _DEFAULTS.items():
        if k not in keep:
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
    df_trace = pd.DataFrame(rows)
    st.dataframe(df_trace, width="stretch", hide_index=True)
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


def _display_frame(df, sql: str = ""):
    """Readable copy of a result: floats to 2 decimals, and a label + one
    measure result without ORDER BY sorted by the measure (largest first)."""
    out = df.copy()
    num = out.select_dtypes("number").columns
    if len(num) == 1 and len(out.columns) > 1 and "order by" not in (sql or "").lower():
        out = out.sort_values(num[0], ascending=False, kind="stable")
    floats = out.select_dtypes("float").columns
    out[floats] = out[floats].round(2)
    return out


def _inline_table(df, sql: str = ""):
    if df is not None and 0 < len(df) <= 50:
        st.dataframe(_display_frame(df, sql), width="stretch", hide_index=True)


def _render_sql_editor(analysis: dict):
    """Show the SQL; let the user edit and re-run it through the same guard."""
    sql = analysis["sql_query"]
    if sql.startswith("--") or st.session_state.db is None:
        st.code(sql, language="sql")
        return
    key = f"sql_{analysis.get('run_id') or id(analysis)}"
    edited = st.text_area("SQL (editable, read-only guard applies)", sql, key=key, height=140)
    if st.button("Run edited SQL", key=f"{key}_run"):
        df, err = st.session_state.db.execute_query(edited)
        if err:
            st.error(err)
        else:
            st.dataframe(df, width="stretch", hide_index=True)


def _render_feedback(analysis: dict):
    run_id = analysis.get("run_id")
    if not run_id:
        return
    rating = st.feedback("thumbs", key=f"fb_{run_id}")
    if rating is not None and not st.session_state.get(f"fb_done_{run_id}"):
        feedback.record(run_id, "streamlit", "up" if rating == 1 else "down", analysis.get("question", ""),
                        analysis.get("sql_query"), st.query_params.get("ds", ""))
        st.session_state[f"fb_done_{run_id}"] = True
        st.toast("Thanks — feedback recorded.")


def render_assistant_body(analysis: dict, should_stream: bool):
    """Render chart + narrative + chips + expanders inside an assistant message."""
    validation = analysis.get("validation") or {}
    conf = validation.get("confidence")

    if analysis.get("declined"):
        # Out-of-scope / write request: a plain decline, no warnings or red confidence.
        st.info(md_safe(analysis.get("narrative", "")))
        _render_feedback(analysis)
        return

    # Narrative
    if analysis.get("narrative"):
        narrative_text = analysis["narrative"]
        if "encountered an error" in narrative_text:
            st.info("The AI narrative service was temporarily unavailable. Your data results are shown below.")
        elif should_stream:
            st.write_stream(md_safe(c) for c in type_stream(narrative_text, words_per_chunk=2, delay=0.025))
        else:
            st.markdown(md_safe(narrative_text))

    # Chart, or the result as a table when no chart fits (e.g. 1-3 rows)
    if analysis.get("chart") is not None:
        st.plotly_chart(analysis["chart"], width="stretch", config={"displayModeBar": False})
    else:
        _inline_table(analysis.get("result_df"), analysis.get("sql_query") or "")

    if analysis.get("prediction_chart") is not None:
        st.plotly_chart(analysis["prediction_chart"], width="stretch", config={"displayModeBar": False})

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

    _render_feedback(analysis)

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
                    st.dataframe(_display_frame(analysis["result_df"], analysis.get("sql_query") or ""),
                                 width="stretch", hide_index=True)
                tab_idx += 1
            if "Query" in tab_names:
                with tabs[tab_idx]:
                    _render_sql_editor(analysis)
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

def _table_name(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0].lower()
    return "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_") or "data"


def _handle_file_upload(uploaded_files):
    """Load one or more files; the first is the primary table, the rest are
    extra tables the Coder can join."""
    frames = {}
    for f in uploaded_files:
        ok, msg = DEMO.upload_allowed(f.size)
        if not ok or f.size > MAX_UPLOAD_BYTES:
            st.error(msg or f"{f.name} is too large ({f.size / 1024 / 1024:.1f} MB).")
            return False
        df = load_file(f)
        if len(df) > MAX_ROWS_INGEST:
            st.warning(f"{f.name} truncated to {MAX_ROWS_INGEST:,} rows (uploaded {len(df):,}).")
            df = df.head(MAX_ROWS_INGEST)
        frames[_table_name(f.name)] = df
    primary, *rest = frames.items()
    with st.spinner("Profiling dataset..."):
        init_database(primary[1], primary[0], original_filename=uploaded_files[0].name,
                      extra_tables=dict(rest))
    log.info("Loaded %d table(s), primary %s rows=%d", len(frames), primary[0], len(primary[1]))
    return True


def _restore_from_url(ds: str) -> bool:
    """A reload or reconnect starts a fresh session; reopen the dataset the URL
    names so the user lands back in their workspace (chat history is lost)."""
    if ds == "chinook":
        init_sample_workspace()
        return True
    for name, path in _SAMPLES.items():
        if ds == name.lower().replace(" ", "_") and os.path.exists(path):
            init_database(pd.read_csv(path), ds, original_filename=os.path.basename(path))
            return True
    if PERSIST:
        match = next((d for d in store.list_datasets(st.session_state.workspace_id) if d.name == ds), None)
        if match and rehydrate_dataset(match.id):
            st.query_params["view"], st.query_params["ds"] = "chat", ds
            return True
    return False


if st.session_state.schema is None and _QP_VIEW == "chat" and _QP_DATASET:
    with st.spinner("Restoring your workspace..."):
        restored = _restore_from_url(_QP_DATASET)
    if restored:
        st.session_state["_restored_notice"] = True
    else:
        st.query_params.clear()

@st.fragment(run_every=60)
def _keepalive():
    """Renders nothing. Its periodic rerun is server->browser traffic that keeps
    the websocket from going idle: in QA, a question typed after ~7 idle
    minutes was lost while the browser silently reconnected."""


_keepalive()

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
        feature_card("shield", "Guarded SQL",
                     "Read-only queries behind an AST allow-list, engine guards and timeouts.")
    with f6:
        feature_card("document", "Exec-ready reports",
                     "One click turns your session into a branded PDF deliverable.")

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # --- Quick start: sample datasets ---
    section_label("Quick start — try a sample")
    if st.button("Chinook music store — 11 related tables (joins)", key="qs_chinook",
                 width="stretch", type="primary"):
        with st.spinner("Opening sample database..."):
            init_sample_workspace()
        st.rerun()
    qs1, qs2, qs3 = st.columns(3, gap="medium")
    for col, (name, path) in zip([qs1, qs2, qs3], _SAMPLES.items()):
        if col.button(name, key=f"qs_{name}", width="stretch"):
            if os.path.exists(path):
                with st.spinner("Loading & profiling..."):
                    _df = pd.read_csv(path)
                    init_database(_df, name.lower().replace(" ", "_"),
                                  original_filename=os.path.basename(path))
                st.rerun()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # --- Upload your own ---
    section_label("Or upload your own files")
    landing_files = st.file_uploader(
        "Upload CSV or Excel — several files become related tables you can join",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="landing_upload",
    )
    # The server accepts up to 25 MB so an oversized file reaches our own check,
    # which shows a visible error (Streamlit's own rejection is a hover tooltip).
    st.caption(f"Up to {DEMO.max_upload_mb if DEMO.enabled else MAX_UPLOAD_BYTES // 2**20} MB per file · CSV, XLSX, XLS")
    if landing_files:
        try:
            if _handle_file_upload(landing_files):
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
            st.image(landing_image, width="stretch")
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
                    edited_df = st.data_editor(df, width="stretch", num_rows="dynamic")
                    if st.button("Use this data", type="primary"):
                        init_database(edited_df, "image_data", original_filename="image_extract")
                        st.rerun()

    # --- Previously loaded datasets ---
    _library = store.list_datasets(st.session_state.workspace_id) if PERSIST else []
    if _library:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        section_label("Recent datasets")
        cols = st.columns(min(len(_library), 3), gap="medium")
        for i, ds in enumerate(_library[:6]):
            with cols[i % 3]:
                if st.button(
                    f"{ds.name.replace('_', ' ').title()}",
                    key=f"lib_{ds.id}",
                    width="stretch",
                ):
                    with st.spinner("Loading dataset..."):
                        rehydrate_dataset(ds.id)
                    st.rerun()


# =====================================================================
# ANALYSIS / CHAT PAGE
# =====================================================================

else:
    schema = st.session_state.schema

    nav_l, nav_c, nav_r = st.columns([1.2, 4, 1.2])
    if nav_l.button("< Back", key="nav_back", width="stretch"):
        clear_session()
        st.rerun()
    _n_tables = len(st.session_state.db.datasource().tables)
    _title = (st.query_params.get("ds") or schema.table_name).replace("_", " ").title()
    _sub = (f"{_n_tables} tables" if _n_tables > 1
            else f"{schema.row_count:,} rows · {schema.column_count} cols")
    nav_c.markdown(
        f'<div class="da-nav-title" style="text-align:center;padding-top:0.25rem;">'
        f'<span style="font-weight:700;color:var(--text);font-size:1.05rem;">{_title}</span>'
        f'<span style="color:var(--text-mute);font-size:0.8rem;margin-left:0.5rem;">{_sub}</span></div>',
        unsafe_allow_html=True,
    )
    if nav_r.button("New chat", key="nav_new", width="stretch"):
        st.session_state.messages = []
        st.session_state.analyses = []
        st.session_state.agent_log = []
        st.rerun()

    # --- Dataset overview (collapsed) ---
    with st.expander("Dataset overview", expanded=False):
        _tables = st.session_state.db.datasource().tables
        if len(_tables) > 1:
            st.caption("Questions can join across these tables.")
            st.dataframe(
                pd.DataFrame([{"table": t.name, "rows": f"{t.row_count:,}", "columns": len(t.columns),
                               "links to": ", ".join(sorted({fk.ref_table for fk in t.foreign_keys}))}
                              for t in _tables]),
                width="stretch", hide_index=True,
            )
        else:
            render_profile(schema)
            st.dataframe(st.session_state.df.head(MAX_DISPLAY_ROWS), width="stretch")

    # --- Report export (above the chat so the fixed input bar can't cover it) ---
    export_slot = st.container()
    if st.session_state.analyses:
        col_a, col_b, _ = export_slot.columns([1, 1, 2])
        report_clicked = col_a.button("Export PDF report", width="stretch")
        md_clicked = col_b.button("Export Markdown", width="stretch")
    else:
        report_clicked = md_clicked = False

    if st.session_state.pop("_restored_notice", False) and not st.session_state.messages:
        st.info("Reopened your dataset after the page reloaded. Earlier answers in this chat were cleared.")

    # --- Bring your own key (public demo) ---
    if DEMO.enabled:
        with st.expander("Use your own Gemini API key (no question limit)", expanded=False):
            st.session_state.user_api_key = st.text_input(
                "Gemini API key", value=st.session_state.user_api_key, type="password",
                help="Used only for your session; never stored. Free at aistudio.google.com/apikey.",
            )

    # --- Session info bar ---
    _cost = _session_cost_summary()
    if _cost["n_runs"] > 0:
        _info_parts = [
            f'<div class="da-info-item">{icon("bolt", 13)} <strong>{_cost["n_runs"]}</strong> {"run" if _cost["n_runs"] == 1 else "runs"}</div>',
            f'<div class="da-info-item">Duration: <strong>{_cost["total_duration_ms"]/1000:.1f}s</strong></div>',
        ]
        _total_tokens = _cost["total_tokens_in"] + _cost["total_tokens_out"]
        if _total_tokens > 0:
            _info_parts.append(f'<div class="da-info-item">Tokens: <strong>{_total_tokens:,}</strong></div>')
        if _cost["total_cost_usd"] > 0:
            _info_parts.append(f'<div class="da-info-item">Cost: <strong>${_cost["total_cost_usd"]:.5f}</strong></div>')
        st.markdown(
            f'<div class="da-info-bar">{"".join(_info_parts)}</div>',
            unsafe_allow_html=True,
        )

    # --- Starter suggestions ---
    if not st.session_state.messages and schema.suggested_questions:
        section_label("Starter questions")
        st.markdown('<div class="da-suggest-anchor"></div>', unsafe_allow_html=True)
        cols = st.columns(3, gap="medium")
        for i, q in enumerate(schema.suggested_questions[:6]):
            if cols[i % 3].button(q, key=f"sq_{i}", width="stretch"):
                st.session_state["_pending_q"] = q
                st.rerun()

    # --- Chat history ---
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar=":material/person:"):
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

    # --- Handle new chat turn ---
    _vkey = _visitor_key()
    _used = max(st.session_state.questions_asked, _visitor_budget().used(_vkey) if _vkey else 0)
    allowed, limit_msg = DEMO.question_allowed(_used, st.session_state.user_api_key)
    if prompt and not allowed:
        st.warning(limit_msg)
        prompt = None

    if prompt:
        log.info("User: %s", prompt[:80])
        st.session_state.messages.append({"role": "user", "content": prompt})
        if not st.session_state.user_api_key:
            st.session_state.questions_asked += 1
            if _vkey:
                _visitor_budget().charge(_vkey)
        # Show the question right away, not only after the 20-40 s run.
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(prompt)
        key_scope = api_key_override(st.session_state.user_api_key)
        key_scope.__enter__()
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
                    display_question=prompt,
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
                    if error_msg.startswith("OutOfScope"):
                        narrative = ("I can't answer that from this data: "
                                     + error_msg.split(":", 1)[1].strip())
                    elif "QuotaExhausted" in error_msg:
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
                    (st.info if error_msg.startswith("OutOfScope") else st.warning)(narrative)
                else:
                    result_summary = storyteller_agent.summarize_result(
                        result_df, pii.detect(st.session_state.db.datasource()))
                    narrative_stream = storyteller_agent.stream_narrative(
                        question=contextual_q,
                        sql_query=result.get("sql_query", "") or "",
                        result_summary=result_summary,
                        chart_description=(result.get("chart_config") or {}).get("chart_type", ""),
                        validation_warnings=validation.get("warnings", []),
                        prediction_info=result.get("prediction"),
                    )
                    # Collect the first chunk to check for errors before displaying
                    first_chunks = []
                    is_error = False
                    for chunk in narrative_stream:
                        first_chunks.append(chunk)
                        full_so_far = "".join(first_chunks)
                        if "encountered an error" in full_so_far:
                            is_error = True
                            break
                        # Once we have enough text to confirm it's real, stream the rest
                        if len(full_so_far) > 50:
                            break

                    if is_error:
                        narrative = "".join(first_chunks)
                        st.info("The AI narrative service is temporarily unavailable. Your data results are shown below.")
                    else:
                        # Stream remaining chunks with the buffered ones prepended
                        import itertools
                        raw: list = []
                        st.write_stream(_escaped(itertools.chain(first_chunks, narrative_stream), raw))
                        narrative = "".join(raw)

                # Show chart inline on first render
                if result.get("chart") is not None:
                    st.plotly_chart(result["chart"], width="stretch", config={"displayModeBar": False})
                elif has_data and val_status != "rejected":
                    _inline_table(result_df, result.get("sql_query") or "")
                if result.get("prediction_chart") is not None:
                    st.plotly_chart(result["prediction_chart"], width="stretch", config={"displayModeBar": False})

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
                "declined": (result.get("error") or "").startswith("OutOfScope"),
                "run_id": result.get("run_id"),
            }
            analysis_id = None
            if st.session_state.dataset_id and PERSIST:
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
                    "The AI service is rate-limiting requests right now. Please wait "
                    "a moment and try again."
                )
            else:
                user_error = f"The agent crew hit an error. Try rephrasing your question."
            log.error("User-facing error: %s | Raw: %s", user_error, err_str[:300])
            st.session_state.messages.append({
                "role": "assistant",
                "question": prompt,
                "error": user_error,
            })
        finally:
            key_scope.__exit__(None, None, None)
        st.rerun()

    # --- Report export handlers ---
    if (report_clicked or md_clicked) and st.session_state.analyses:
        with export_slot, st.spinner("Assembling report..."):
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
