"""Chart type configurations for the Visualizer Agent."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from enum import Enum


class ChartType(str, Enum):
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    GROUPED_BAR = "grouped_bar"
    KPI_CARD = "kpi_card"
    TABLE = "table"
    FORECAST = "forecast"
    CLUSTER_SCATTER = "cluster_scatter"
    ANOMALY = "anomaly"


class ChartConfig(BaseModel):
    chart_type: ChartType
    title: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    template: str = "dataagent"
