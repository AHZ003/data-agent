"""Register a custom Plotly template matching the DataAgent dark palette."""

import plotly.graph_objects as go
import plotly.io as pio


TEMPLATE_NAME = "dataagent"

_COLORWAY = [
    "#7C5CFF",
    "#34D399",
    "#F59E0B",
    "#60A5FA",
    "#F472B6",
    "#A78BFA",
    "#FBBF24",
    "#4ADE80",
]


def register():
    """Register the dataagent template with Plotly as the default."""
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, -apple-system, sans-serif", color="#E6E6EC", size=13),
            paper_bgcolor="#14141B",
            plot_bgcolor="#14141B",
            colorway=_COLORWAY,
            title=dict(
                font=dict(size=15, color="#E6E6EC", family="Inter"),
                x=0.02,
                xanchor="left",
                pad=dict(t=12, b=8),
            ),
            xaxis=dict(
                gridcolor="#24242F",
                linecolor="#2E2E3C",
                zerolinecolor="#24242F",
                tickcolor="#2E2E3C",
                tickfont=dict(color="#9A9AAD", size=11),
                title=dict(font=dict(color="#9A9AAD", size=12)),
            ),
            yaxis=dict(
                gridcolor="#24242F",
                linecolor="#2E2E3C",
                zerolinecolor="#24242F",
                tickcolor="#2E2E3C",
                tickfont=dict(color="#9A9AAD", size=11),
                title=dict(font=dict(color="#9A9AAD", size=12)),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9A9AAD", size=11),
                bordercolor="#24242F",
            ),
            margin=dict(l=50, r=20, t=50, b=40),
            hoverlabel=dict(
                bgcolor="#1A1A23",
                bordercolor="#2E2E3C",
                font=dict(color="#E6E6EC", family="Inter"),
            ),
        )
    )
    pio.templates[TEMPLATE_NAME] = template
    pio.templates.default = TEMPLATE_NAME
