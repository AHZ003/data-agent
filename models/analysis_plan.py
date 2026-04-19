"""Pydantic models for structured agent inputs and outputs."""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# --- Schema Agent Models ---

class ColumnRole(str, Enum):
    DIMENSION = "dimension"
    MEASURE = "measure"
    TIME_AXIS = "time_axis"
    IDENTIFIER = "identifier"
    TEXT = "text"


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    role: ColumnRole
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    # Numeric stats
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    median_val: Optional[float] = None
    std_val: Optional[float] = None
    # Categorical stats
    top_values: Optional[List[str]] = None
    # Datetime stats
    date_range: Optional[str] = None
    date_frequency: Optional[str] = None


class SemanticSchema(BaseModel):
    table_name: str
    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    suggested_analyses: List[str] = Field(default_factory=list)
    suggested_questions: List[str] = Field(default_factory=list)


# --- Planner Agent Models ---

class AnalysisType(str, Enum):
    DESCRIPTIVE = "descriptive"
    TREND = "trend"
    COMPARISON = "comparison"
    CORRELATION = "correlation"
    PREDICTION = "prediction"
    SEGMENTATION = "segmentation"
    ANOMALY = "anomaly"
    REPORT = "report"


class PlanStep(BaseModel):
    agent: str
    task: str
    depends_on: Optional[int] = None  # index of step this depends on


class AnalysisPlan(BaseModel):
    question: str
    analysis_types: List[AnalysisType]
    steps: List[PlanStep]


# --- Coder Agent Models ---

class CodeResult(BaseModel):
    sql_query: str
    success: bool
    error: Optional[str] = None
    row_count: int = 0
    columns: List[str] = Field(default_factory=list)


# --- Critic Agent Models ---

class ValidationStatus(str, Enum):
    VALIDATED = "validated"
    VALIDATED_WITH_WARNINGS = "validated_with_warnings"
    REJECTED = "rejected"


class ValidationReport(BaseModel):
    status: ValidationStatus
    confidence: int = Field(ge=0, le=100)
    warnings: List[str] = Field(default_factory=list)
    corrections_made: List[str] = Field(default_factory=list)
    rejection_reason: Optional[str] = None


# --- Agent Activity Log ---

class AgentLogEntry(BaseModel):
    agent_name: str
    task: str
    input_summary: str
    output_summary: str
    duration_seconds: float
    status: str = "success"
