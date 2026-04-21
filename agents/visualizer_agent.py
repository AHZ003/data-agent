"""Visualizer Agent - Chart selection and generation using Plotly."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

from models.chart_config import ChartType, ChartConfig
from assets.plotly_theme import register as _register_plotly_theme

_register_plotly_theme()


def _classify_chart_type(df: pd.DataFrame, question: str) -> ChartConfig:
    """Determine the best chart type based on data shape and question."""
    if df is None or len(df) == 0:
        return ChartConfig(chart_type=ChartType.TABLE, title=question)

    question_lower = question.lower()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    # Check for string columns that look like dates — build new lists, don't mutate
    detected_dates = []
    for col in cat_cols:
        try:
            pd.to_datetime(df[col].head(10))
            detected_dates.append(col)
        except (ValueError, TypeError):
            pass
    if detected_dates:
        date_cols = date_cols + detected_dates
        cat_cols = [c for c in cat_cols if c not in detected_dates]

    # Single value -> KPI card
    if len(df) == 1 and len(num_cols) == 1:
        return ChartConfig(
            chart_type=ChartType.KPI_CARD,
            title=question,
            y_column=num_cols[0],
        )

    # Trivial results (very few rows, no explicit chart request) -> table
    # A chart with 1-3 data points is usually less useful than the raw numbers.
    _chart_keywords = [
        "distribution", "histogram", "chart", "plot", "graph", "visuali",
        "proportion", "share", "percentage", "pie", "trend", "over time",
    ]
    user_wants_chart = any(w in question_lower for w in _chart_keywords)
    if not user_wants_chart and len(df) <= 3 and len(df.columns) <= 4:
        return ChartConfig(chart_type=ChartType.TABLE, title=question)

    # No usable columns -> table
    if not num_cols and not cat_cols and not date_cols:
        return ChartConfig(chart_type=ChartType.TABLE, title=question)

    # Keywords override
    if any(w in question_lower for w in ["distribution", "histogram"]):
        col = num_cols[0] if num_cols else df.columns[0]
        return ChartConfig(
            chart_type=ChartType.HISTOGRAM,
            title=question,
            x_column=col,
            x_label=col,
            y_label="Count",
        )
    if any(w in question_lower for w in ["proportion", "share", "percentage", "pie"]):
        if cat_cols and num_cols and df[cat_cols[0]].nunique() <= 8:
            return ChartConfig(
                chart_type=ChartType.PIE,
                title=question,
                x_column=cat_cols[0],
                y_column=num_cols[0],
            )

    # Datetime + numeric -> line chart
    if date_cols and num_cols:
        return ChartConfig(
            chart_type=ChartType.LINE,
            title=question,
            x_column=date_cols[0],
            y_column=num_cols[0],
            x_label=date_cols[0],
            y_label=num_cols[0],
        )

    # Categorical + numeric
    if cat_cols and num_cols:
        n_categories = df[cat_cols[0]].nunique()
        if len(num_cols) > 1:
            return ChartConfig(
                chart_type=ChartType.GROUPED_BAR,
                title=question,
                x_column=cat_cols[0],
                y_column=num_cols[0],
                color_column=cat_cols[0] if len(cat_cols) > 1 else None,
                x_label=cat_cols[0],
                y_label=num_cols[0],
            )
        if n_categories > 7:
            return ChartConfig(
                chart_type=ChartType.HORIZONTAL_BAR,
                title=question,
                x_column=num_cols[0],
                y_column=cat_cols[0],
                x_label=num_cols[0],
                y_label=cat_cols[0],
            )
        return ChartConfig(
            chart_type=ChartType.BAR,
            title=question,
            x_column=cat_cols[0],
            y_column=num_cols[0],
            x_label=cat_cols[0],
            y_label=num_cols[0],
        )

    # Two numeric -> scatter
    if len(num_cols) >= 2:
        return ChartConfig(
            chart_type=ChartType.SCATTER,
            title=question,
            x_column=num_cols[0],
            y_column=num_cols[1],
            x_label=num_cols[0],
            y_label=num_cols[1],
        )

    # Default to table
    return ChartConfig(chart_type=ChartType.TABLE, title=question)


def generate_chart(
    df: pd.DataFrame,
    question: str,
    config: Optional[ChartConfig] = None,
) -> Optional[go.Figure]:
    """
    Generate a Plotly chart based on the data and question.

    Returns a Plotly Figure or None if table display is best.
    """
    if df is None or len(df) == 0:
        return None

    if config is None:
        config = _classify_chart_type(df, question)

    template = config.template

    try:
        if config.chart_type == ChartType.KPI_CARD:
            value = df[config.y_column].iloc[0] if config.y_column else df.iloc[0, 0]
            fig = go.Figure(
                go.Indicator(mode="number", value=float(value), title={"text": config.title})
            )
            fig.update_layout(template=template, height=300)
            return fig

        if config.chart_type == ChartType.BAR:
            fig = px.bar(
                df, x=config.x_column, y=config.y_column,
                title=config.title, template=template,
                labels={config.x_column: config.x_label, config.y_column: config.y_label},
            )
            return fig

        if config.chart_type == ChartType.HORIZONTAL_BAR:
            fig = px.bar(
                df, x=config.y_column, y=config.x_column, orientation="h",
                title=config.title, template=template,
            )
            return fig

        if config.chart_type == ChartType.LINE:
            fig = px.line(
                df, x=config.x_column, y=config.y_column,
                title=config.title, template=template,
                labels={config.x_column: config.x_label, config.y_column: config.y_label},
            )
            return fig

        if config.chart_type == ChartType.SCATTER:
            fig = px.scatter(
                df, x=config.x_column, y=config.y_column,
                title=config.title, template=template,
                labels={config.x_column: config.x_label, config.y_column: config.y_label},
            )
            return fig

        if config.chart_type == ChartType.PIE:
            fig = px.pie(
                df, names=config.x_column, values=config.y_column,
                title=config.title, template=template,
            )
            return fig

        if config.chart_type == ChartType.HISTOGRAM:
            fig = px.histogram(
                df, x=config.x_column,
                title=config.title, template=template,
                labels={config.x_column: config.x_label},
            )
            return fig

        if config.chart_type == ChartType.GROUPED_BAR:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if len(num_cols) >= 2:
                fig = go.Figure()
                for col in num_cols[:4]:
                    fig.add_trace(go.Bar(name=col, x=df[config.x_column], y=df[col]))
                fig.update_layout(
                    barmode="group", title=config.title, template=template,
                )
                return fig
            return px.bar(
                df, x=config.x_column, y=config.y_column,
                title=config.title, template=template,
            )
    except Exception:
        return None

    # TABLE or unknown -> return None (caller displays as dataframe)
    return None


def get_chart_config(df: pd.DataFrame, question: str) -> ChartConfig:
    """Public method to get chart configuration without generating the chart."""
    return _classify_chart_type(df, question)
