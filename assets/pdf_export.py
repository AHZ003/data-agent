"""Branded PDF report builder using ReportLab."""

from datetime import datetime
from io import BytesIO
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    PageBreak, NextPageTemplate, Flowable,
)

from models.report import AnalysisReport


# Brand palette (matches app light theme)
BRAND_ACCENT = colors.HexColor("#6C5CE7")
BRAND_ACCENT_DARK = colors.HexColor("#5A4BD1")
TEXT_PRIMARY = colors.HexColor("#1A1D26")
TEXT_SECONDARY = colors.HexColor("#5A6070")
TEXT_MUTED = colors.HexColor("#8E95A3")
BG_PANEL = colors.HexColor("#F7F8FA")
BORDER = colors.HexColor("#E2E5EA")


def _make_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "cover_eyebrow": ParagraphStyle(
            "CoverEyebrow", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=9, leading=11,
            textColor=BRAND_ACCENT, alignment=TA_LEFT, spaceAfter=14,
            letterSpacing=2,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=base["Title"],
            fontName="Helvetica-Bold", fontSize=36, leading=42,
            textColor=TEXT_PRIMARY, alignment=TA_LEFT, spaceAfter=16,
        ),
        "cover_sub": ParagraphStyle(
            "CoverSub", parent=base["Normal"],
            fontName="Helvetica", fontSize=13, leading=19,
            textColor=TEXT_SECONDARY, alignment=TA_LEFT, spaceAfter=8,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta", parent=base["Normal"],
            fontName="Helvetica", fontSize=9, leading=12,
            textColor=TEXT_MUTED, alignment=TA_LEFT,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"],
            fontName="Helvetica-Bold", fontSize=20, leading=26,
            textColor=TEXT_PRIMARY, spaceBefore=4, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"],
            fontName="Helvetica-Bold", fontSize=13, leading=18,
            textColor=TEXT_PRIMARY, spaceBefore=12, spaceAfter=6,
        ),
        "eyebrow": ParagraphStyle(
            "Eyebrow", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=8, leading=10,
            textColor=BRAND_ACCENT, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontName="Helvetica", fontSize=10.5, leading=16,
            textColor=TEXT_PRIMARY, alignment=TA_JUSTIFY, spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontName="Helvetica", fontSize=10.5, leading=16,
            textColor=TEXT_PRIMARY, leftIndent=14, bulletIndent=4,
            spaceAfter=4,
        ),
        "toc_item": ParagraphStyle(
            "TocItem", parent=base["Normal"],
            fontName="Helvetica", fontSize=11, leading=20,
            textColor=TEXT_PRIMARY,
        ),
    }


class BrandMark(Flowable):
    """A simple violet rounded square + wordmark."""

    def __init__(self, size: float = 9 * mm, label: str = "DataAgent"):
        super().__init__()
        self.size = size
        self.label = label
        self.width = 60 * mm
        self.height = size

    def draw(self):
        c: Canvas = self.canv
        c.saveState()
        c.setFillColor(BRAND_ACCENT)
        c.roundRect(0, 0, self.size, self.size, 2 * mm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", self.size * 0.55)
        c.drawCentredString(self.size / 2, self.size * 0.28, "✦")
        c.setFillColor(TEXT_PRIMARY)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(self.size + 3 * mm, self.size * 0.28, self.label)
        c.restoreState()


def _draw_page_chrome(canvas: Canvas, doc, is_cover: bool = False):
    canvas.saveState()
    if is_cover:
        # Accent bar on left edge of cover
        canvas.setFillColor(BRAND_ACCENT)
        canvas.rect(0, 0, 6 * mm, A4[1], stroke=0, fill=1)
    else:
        # Header rule
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.4)
        canvas.line(20 * mm, A4[1] - 15 * mm, A4[0] - 20 * mm, A4[1] - 15 * mm)
        # Brand mark in header
        canvas.setFillColor(BRAND_ACCENT)
        canvas.roundRect(20 * mm, A4[1] - 13 * mm, 4 * mm, 4 * mm, 0.8 * mm, stroke=0, fill=1)
        canvas.setFillColor(TEXT_SECONDARY)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(25.5 * mm, A4[1] - 12 * mm, "DataAgent")
        canvas.setFillColor(TEXT_MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(
            A4[0] - 20 * mm, A4[1] - 12 * mm,
            f"Analysis Report · Page {doc.page - 1}",
        )
        # Footer
        canvas.setFillColor(TEXT_MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(
            20 * mm, 12 * mm,
            f"Generated {datetime.now().strftime('%B %d, %Y')}",
        )
        canvas.drawRightString(
            A4[0] - 20 * mm, 12 * mm,
            "Confidential · AI-generated",
        )
    canvas.restoreState()


def _cover_page(canvas: Canvas, doc):
    _draw_page_chrome(canvas, doc, is_cover=True)


def _content_page(canvas: Canvas, doc):
    _draw_page_chrome(canvas, doc, is_cover=False)


def build_report_pdf(report: AnalysisReport, dataset_name: Optional[str] = None) -> bytes:
    """Render an AnalysisReport to a styled PDF. Returns raw bytes."""
    buf = BytesIO()
    styles = _make_styles()

    margin = 20 * mm
    page_w, page_h = A4

    cover_frame = Frame(
        margin + 6 * mm, margin, page_w - margin * 2 - 6 * mm, page_h - margin * 2,
        leftPadding=0, rightPadding=0, topPadding=40 * mm, bottomPadding=0,
        id="cover",
    )
    content_frame = Frame(
        margin, margin + 8 * mm, page_w - margin * 2, page_h - margin * 2 - 20 * mm,
        leftPadding=0, rightPadding=0, topPadding=6 * mm, bottomPadding=0,
        id="content",
    )

    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
        title=report.title,
        author="DataAgent",
    )
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[cover_frame], onPage=_cover_page),
        PageTemplate(id="Content", frames=[content_frame], onPage=_content_page),
    ])

    story = []

    # --- Cover ---
    story.append(BrandMark())
    story.append(Spacer(1, 30 * mm))
    story.append(Paragraph("ANALYSIS REPORT", styles["cover_eyebrow"]))
    story.append(Paragraph(report.title, styles["cover_title"]))
    story.append(Paragraph(
        report.executive_summary or "Executive summary not available.",
        styles["cover_sub"],
    ))
    story.append(Spacer(1, 20 * mm))
    meta_lines = [
        f"Prepared by&nbsp;&nbsp;·&nbsp;&nbsp;DataAgent Multi-Agent System",
        f"Generated&nbsp;&nbsp;·&nbsp;&nbsp;{datetime.now().strftime('%B %d, %Y')}",
    ]
    if dataset_name:
        meta_lines.insert(0, f"Dataset&nbsp;&nbsp;·&nbsp;&nbsp;{dataset_name}")
    for line in meta_lines:
        story.append(Paragraph(line, styles["cover_meta"]))
        story.append(Spacer(1, 2 * mm))

    story.append(NextPageTemplate("Content"))
    story.append(PageBreak())

    # --- Table of contents ---
    story.append(Paragraph("CONTENTS", styles["eyebrow"]))
    story.append(Paragraph("Table of contents", styles["h1"]))
    story.append(Spacer(1, 4 * mm))
    toc_items = [
        "1.&nbsp;&nbsp;&nbsp;Executive summary",
        "2.&nbsp;&nbsp;&nbsp;Data overview",
        "3.&nbsp;&nbsp;&nbsp;Key findings",
    ]
    if report.predictions:
        toc_items.append(f"{len(toc_items)+1}.&nbsp;&nbsp;&nbsp;Predictions")
    if report.caveats:
        toc_items.append(f"{len(toc_items)+1}.&nbsp;&nbsp;&nbsp;Caveats & limitations")
    if report.next_questions:
        toc_items.append(f"{len(toc_items)+1}.&nbsp;&nbsp;&nbsp;Suggested next questions")
    for item in toc_items:
        story.append(Paragraph(item, styles["toc_item"]))

    story.append(PageBreak())

    # --- Executive summary ---
    story.append(Paragraph("SECTION 1", styles["eyebrow"]))
    story.append(Paragraph("Executive summary", styles["h1"]))
    story.append(Paragraph(report.executive_summary, styles["body"]))
    story.append(Spacer(1, 6 * mm))

    # --- Data overview ---
    story.append(Paragraph("SECTION 2", styles["eyebrow"]))
    story.append(Paragraph("Data overview", styles["h1"]))
    story.append(Paragraph(report.data_overview or "—", styles["body"]))
    story.append(Spacer(1, 6 * mm))

    # --- Findings ---
    story.append(Paragraph("SECTION 3", styles["eyebrow"]))
    story.append(Paragraph("Key findings", styles["h1"]))
    for i, finding in enumerate(report.findings, 1):
        story.append(Paragraph(f"{i:02d}. {finding.title}", styles["h2"]))
        story.append(Paragraph(finding.content, styles["body"]))
        story.append(Spacer(1, 3 * mm))

    # --- Predictions ---
    if report.predictions:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("SECTION 4", styles["eyebrow"]))
        story.append(Paragraph("Predictions", styles["h1"]))
        story.append(Paragraph(str(report.predictions), styles["body"]))

    # --- Caveats ---
    if report.caveats:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("CAVEATS", styles["eyebrow"]))
        story.append(Paragraph("Caveats & limitations", styles["h1"]))
        for c in report.caveats:
            story.append(Paragraph(f"•&nbsp;&nbsp;{c}", styles["bullet"]))

    # --- Next questions ---
    if report.next_questions:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("WHAT'S NEXT", styles["eyebrow"]))
        story.append(Paragraph("Suggested next questions", styles["h1"]))
        for q in report.next_questions:
            story.append(Paragraph(f"•&nbsp;&nbsp;{q}", styles["bullet"]))

    doc.build(story)
    return buf.getvalue()
