"""CSV/Excel/Image upload and parsing."""

import pandas as pd
import io
from typing import Optional


def load_csv(file) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file)


def load_excel(file) -> pd.DataFrame:
    """Load an Excel file into a DataFrame."""
    return pd.read_excel(file, engine="openpyxl")


def load_file(file) -> pd.DataFrame:
    """Load a file based on its extension."""
    filename = file.name.lower()
    if filename.endswith(".csv"):
        return load_csv(file)
    elif filename.endswith((".xlsx", ".xls")):
        return load_excel(file)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def parse_csv_string(csv_string: str) -> Optional[pd.DataFrame]:
    """Parse a CSV string (from vision extraction) into a DataFrame."""
    try:
        return pd.read_csv(io.StringIO(csv_string))
    except Exception:
        return None


def get_basic_info(df: pd.DataFrame) -> dict:
    """Get basic information about a DataFrame."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": [
            {"name": col, "dtype": str(df[col].dtype)} for col in df.columns
        ],
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }
