"""End-to-end integration tests (require API key)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from core.database import Database
from agents.schema_agent import profile_dataframe
from agents.visualizer_agent import generate_chart, get_chart_config
from agents.predictor_agent import detect_anomalies, cluster_data
from models.chart_config import ChartType


def _make_test_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Revenue": np.random.normal(500, 100, 365).cumsum(),
        "Category": np.random.choice(["Electronics", "Clothing", "Food"], 365),
        "Quantity": np.random.randint(1, 100, 365),
    })


def test_visualizer_bar_chart():
    df = pd.DataFrame({"Category": ["A", "B", "C"], "Revenue": [100, 200, 300]})
    config = get_chart_config(df, "Revenue by category")
    assert config.chart_type in (ChartType.BAR, ChartType.HORIZONTAL_BAR)
    chart = generate_chart(df, "Revenue by category", config)
    assert chart is not None


def test_visualizer_line_chart():
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=30),
        "Revenue": range(30),
    })
    config = get_chart_config(df, "Revenue trend over time")
    assert config.chart_type == ChartType.LINE


def test_visualizer_kpi_card():
    df = pd.DataFrame({"Total": [42000]})
    config = get_chart_config(df, "What is total revenue?")
    assert config.chart_type == ChartType.KPI_CARD


def test_anomaly_detection():
    df = _make_test_data()
    # Add some anomalies
    df.loc[10, "Quantity"] = 999
    df.loc[50, "Quantity"] = -50
    anomaly_df, fig, metrics = detect_anomalies(df, numeric_cols=["Quantity"])
    assert metrics["total_anomalies"] > 0
    assert fig is not None


def test_clustering():
    np.random.seed(42)
    df = pd.DataFrame({
        "Feature1": np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)]),
        "Feature2": np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)]),
    })
    clustered_df, fig, metrics = cluster_data(df)
    assert "Cluster" in clustered_df.columns
    assert metrics["optimal_k"] >= 2
    assert fig is not None


if __name__ == "__main__":
    test_visualizer_bar_chart()
    print("✓ test_visualizer_bar_chart passed")
    test_visualizer_line_chart()
    print("✓ test_visualizer_line_chart passed")
    test_visualizer_kpi_card()
    print("✓ test_visualizer_kpi_card passed")
    test_anomaly_detection()
    print("✓ test_anomaly_detection passed")
    test_clustering()
    print("✓ test_clustering passed")
    print("\nAll end-to-end tests passed!")
