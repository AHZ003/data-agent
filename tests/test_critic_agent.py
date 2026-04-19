"""Tests for the Critic Agent."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from agents.critic_agent import validate_sql_result, validate_statistical_claim, validate_chart
from models.analysis_plan import SemanticSchema, ColumnProfile, ColumnRole, ValidationStatus
from models.chart_config import ChartConfig, ChartType


def _sample_schema():
    return SemanticSchema(
        table_name="test",
        row_count=100,
        column_count=3,
        columns=[
            ColumnProfile(name="Category", dtype="object", role=ColumnRole.DIMENSION),
            ColumnProfile(name="Revenue", dtype="float64", role=ColumnRole.MEASURE),
        ],
    )


def test_validate_empty_result():
    report = validate_sql_result(None, "SELECT ...", "test?", _sample_schema())
    assert report.status == ValidationStatus.REJECTED
    assert report.confidence < 40


def test_validate_normal_result():
    df = pd.DataFrame({"Category": ["A", "B"], "Revenue": [100, 200]})
    report = validate_sql_result(df, "SELECT ...", "test?", _sample_schema())
    assert report.status == ValidationStatus.VALIDATED
    assert report.confidence >= 70


def test_validate_small_sample():
    report = validate_statistical_claim(sample_size=5)
    assert len(report.warnings) > 0
    assert any("sample" in w.lower() for w in report.warnings)


def test_validate_weak_correlation():
    report = validate_statistical_claim(sample_size=100, correlation=0.05)
    assert any("negligible" in w.lower() for w in report.warnings)


def test_validate_pie_chart_too_many():
    df = pd.DataFrame({
        "Category": [f"Cat_{i}" for i in range(20)],
        "Value": range(20),
    })
    config = ChartConfig(
        chart_type=ChartType.PIE,
        title="Test",
        x_column="Category",
        y_column="Value",
    )
    report = validate_chart(df, config)
    assert len(report.warnings) > 0


if __name__ == "__main__":
    test_validate_empty_result()
    print("✓ test_validate_empty_result passed")
    test_validate_normal_result()
    print("✓ test_validate_normal_result passed")
    test_validate_small_sample()
    print("✓ test_validate_small_sample passed")
    test_validate_weak_correlation()
    print("✓ test_validate_weak_correlation passed")
    test_validate_pie_chart_too_many()
    print("✓ test_validate_pie_chart_too_many passed")
    print("\nAll critic agent tests passed!")
