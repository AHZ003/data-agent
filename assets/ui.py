"""UI primitives: SVG icons, cards, theme injection."""

from pathlib import Path
import streamlit as st

_ICONS = {
    "sparkles": '<path d="M12 3l1.9 4.6L18.5 9.5l-4.6 1.9L12 16l-1.9-4.6L5.5 9.5l4.6-1.9z"/><path d="M19 14l.9 2.1L22 17l-2.1.9L19 20l-.9-2.1L16 17l2.1-.9z"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v6c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 11v6c0 1.7 4 3 9 3s9-1.3 9-3v-6"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M7 15l4-4 4 4 5-6"/>',
    "predict": '<path d="M3 12h3l3-9 4 18 3-9h5"/>',
    "shield": '<path d="M12 2l8 4v6c0 5-3.5 9.5-8 10-4.5-.5-8-5-8-10V6l8-4z"/><path d="M9 12l2 2 4-4"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5"/><path d="M12 3v12"/>',
    "play": '<path d="M5 3l14 9-14 9V3z"/>',
    "document": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="M21 21l-4.3-4.3"/>',
    "bolt": '<path d="M13 2L3 14h9l-1 8 10-12h-9z"/>',
    "check": '<path d="M20 6L9 17l-5-5"/>',
    "alert": '<path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.3 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>',
    "x": '<path d="M18 6L6 18"/><path d="M6 6l12 12"/>',
    "brain": '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2z"/>',
    "layers": '<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>',
    "pen": '<path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5z"/>',
}


def icon(name: str, size: int = 16, color: str = "currentColor") -> str:
    """Return inline SVG string for an icon."""
    body = _ICONS.get(name, _ICONS["sparkles"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )


def inject_theme():
    """Inject the global CSS stylesheet into the Streamlit app."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    # Force sidebar open via JS — Streamlit Cloud sometimes ignores initial_sidebar_state
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
        (function() {
            var parent = window.parent.document;
            var sidebar = parent.querySelector('section[data-testid="stSidebar"]');
            if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
                var btn = parent.querySelector('[data-testid="collapsedControl"]')
                         || parent.querySelector('[data-testid="stSidebarCollapsedControl"]')
                         || parent.querySelector('button[kind="header"]');
                if (btn) { setTimeout(function(){ btn.click(); }, 500); }
            }
        })();
        </script>
        """,
        height=0,
    )


def brand():
    """Render the sidebar brand block."""
    st.markdown(
        f"""
        <div class="da-brand">
            <span class="da-brand-mark">{icon("sparkles", 16, "#fff")}</span>
            <span class="da-brand-name">DataAgent</span>
        </div>
        <div style="color:var(--text-mute);font-size:0.75rem;margin-bottom:1rem;">
            AI-native analytics workspace
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_label(text: str):
    st.markdown(f'<div class="da-sidebar-label">{text}</div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "AI Data Analyst"):
    st.markdown(
        f"""
        <div class="da-hero">
            <div class="da-hero-eyebrow">{icon("bolt", 12, "#B8A5FF")} {eyebrow}</div>
            <h1 class="da-hero-title">{title}</h1>
            <p class="da-hero-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon_name: str, title: str, desc: str):
    st.markdown(
        f"""
        <div class="da-feature">
            <div class="da-feature-icon">{icon(icon_name, 18)}</div>
            <div class="da-feature-title">{title}</div>
            <p class="da-feature-desc">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_header(question: str):
    st.markdown(
        f"""
        <div style="margin: 0 0 0.5rem 0;">
            <div class="da-result-q">Question</div>
            <div class="da-result-title">{question}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def narrative_block(text: str):
    st.markdown(f'<div class="da-result-narrative">{text}</div>', unsafe_allow_html=True)


def chip(text: str, kind: str = ""):
    """kind: '', 'ok', 'warn', 'err'"""
    cls = f"da-chip {kind}".strip()
    return f'<span class="{cls}">{text}</span>'
