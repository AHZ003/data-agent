"""Report structure models."""

from typing import List, Optional, Any
from pydantic import BaseModel


class ReportSection(BaseModel):
    title: str
    content: str
    chart_data: Optional[Any] = None


class AnalysisReport(BaseModel):
    title: str
    executive_summary: str
    data_overview: str
    findings: List[ReportSection]
    predictions: Optional[str] = None
    caveats: List[str]
    next_questions: List[str]
