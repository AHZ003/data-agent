"""Register a custom Plotly template matching the DataAgent light palette."""

import plotly.graph_objects as go
import plotly.io as pio


TEMPLATE_NAME = "dataagent"

_COLORWAY = [
    "#6C5CE7",
    "#00B894",
    "#FDCB6E",
    "#0984E3",
    "#E17055",
    "#A29BFE",
    "#00CEC9",
    "#FAB1A0",
]


def register():
    """Register the dataagent template with Plotly as the default."""
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, -apple-system, sans-serif", color="#1A1D26", size=13),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            colorway=_COLORWAY,
            title=dict(
                font=dict(size=15, color="#1A1D26", family="Inter"),
                x=0.02,
                xanchor="left",
                pad=dict(t=12, b=8),
            ),
            xaxis=dict(
                gridcolor="#E2E5EA",
                linecolor="#D0D4DB",
                zerolinecolor="#E2E5EA",
                tickcolor="#D0D4DB",
                tickfont=dict(color="#5A6070", size=11),
                title=dict(font=dict(color="#5A6070", size=12)),
            ),
            yaxis=dict(
                gridcolor="#E2E5EA",
                linecolor="#D0D4DB",
                zerolinecolor="#E2E5EA",
                tickcolor="#D0D4DB",
                tickfont=dict(color="#5A6070", size=11),
                title=dict(font=dict(color="#5A6070", size=12)),
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)",
                font=dict(color="#5A6070", size=11),
                bordercolor="#E2E5EA",
            ),
            margin=dict(l=50, r=20, t=50, b=40),
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                bordercolor="#D0D4DB",
                font=dict(color="#1A1D26", family="Inter"),
            ),
        )
    )
    pio.templates[TEMPLATE_NAME] = template
    pio.templates.default = TEMPLATE_NAME
