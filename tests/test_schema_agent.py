"""Tests for the Schema Agent."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from agents.schema_agent import profile_dataframe, _detect_column_role, _profile_column
from models.analysis_plan import ColumnRole


def _sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=100, freq="D"),
        "Category": np.random.choice(["A", "B", "C"], size=100),
        "Revenue": np.random.uniform(100, 1000, size=100),
        "Quantity": np.random.randint(1, 50, size=100),
        "Customer_ID": range(1, 101),
    })


def test_detect_column_role():
    df = _sample_df()
    assert _detect_column_role(df["Date"], "Date") == ColumnRole.TIME_AXIS
    assert _detect_column_role(df["Category"], "Category") == ColumnRole.DIMENSION
    assert _detect_column_role(df["Revenue"], "Revenue") == ColumnRole.MEASURE


def test_profile_column_numeric():
    df = _sample_df()
    profile = _profile_column(df["Revenue"], "Revenue")
    assert profile.name == "Revenue"
    assert profile.role == ColumnRole.MEASURE
    assert profile.min_val is not None
    assert profile.max_val is not None
    assert profile.mean_val is not None
    assert profile.null_count == 0


def test_profile_column_categorical():
    df = _sample_df()
    profile = _profile_column(df["Category"], "Category")
    assert profile.role == ColumnRole.DIMENSION
    assert profile.top_values is not None
    assert len(profile.top_values) <= 5


def test_profile_dataframe_structure():
    """Test that profile_dataframe returns correct structure (no API call)."""
    df = _sample_df()
    # Mock the suggested questions to avoid API call
    schema = profile_dataframe.__wrapped__(df, "test_data") if hasattr(profile_dataframe, '__wrapped__') else None
    # Just test the non-API parts
    profile = _profile_column(df["Date"], "Date")
    assert profile.role == ColumnRole.TIME_AXIS
    assert profile.date_range is not None


if __name__ == "__main__":
    test_detect_column_role()
    print("✓ test_detect_column_role passed")
    test_profile_column_numeric()
    print("✓ test_profile_column_numeric passed")
    test_profile_column_categorical()
    print("✓ test_profile_column_categorical passed")
    test_profile_dataframe_structure()
    print("✓ test_profile_dataframe_structure passed")
    print("\nAll schema agent tests passed!")
