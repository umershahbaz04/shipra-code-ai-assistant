from __future__ import annotations

import re
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _value(row: dict[str, Any], *keys: str, default: str = "-") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _font_name() -> str:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            try:
                pdfmetrics.registerFont(TTFont("ShipraUnicode", candidate))
                return "ShipraUnicode"
            except Exception:
                pass
    return "Helvetica"


def safe_pdf_filename(title: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_")
    return f"{clean[:80] or 'shipra_orders'}.pdf"


def build_orders_pdf(
    groups: list[dict[str, Any]],
    *,
    title: str,
    filter_text: str,
) -> bytes:
    """Create a downloadable PDF from already validated Shipra order rows."""
    buffer = BytesIO()
    font = _font_name()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ShipraTitle",
        parent=styles["Title"],
        fontName=font,
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ShipraBody",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=8,
        leading=10,
    )
    heading_style = ParagraphStyle(
        "ShipraHeading",
        parent=styles["Heading2"],
        fontName=font,
        fontSize=12,
        leading=15,
        spaceBefore=8,
        spaceAfter=6,
    )

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=10 * mm,
        leftMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title=title,
        author="Shipra AI Assistant",
    )

    total_orders = sum(len(group.get("rows") or []) for group in groups)
    story = [
        Paragraph(escape(title), title_style),
        Paragraph(escape(f"Filter: {filter_text}"), body_style),
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
            f"Total orders: {total_orders}",
            body_style,
        ),
        Spacer(1, 5 * mm),
    ]

    for group_index, group in enumerate(groups):
        group_name = str(group.get("name") or "Orders")
        rows = group.get("rows") or []
        story.append(
            Paragraph(
                escape(f"Store: {group_name} ({len(rows)} orders)"),
                heading_style,
            )
        )

        table_rows: list[list[Any]] = [
            ["#", "Order No", "Order Date", "Amount", "Tracking Status"]
        ]
        for index, row in enumerate(rows, 1):
            table_rows.append(
                [
                    str(index),
                    Paragraph(escape(_value(row, "OrderNo", "orderNo", "RefNo", "refNo")), body_style),
                    Paragraph(escape(_value(row, "OrderDate", "orderDate")), body_style),
                    Paragraph(escape(_value(row, "Amount", "amount")), body_style),
                    Paragraph(
                        escape(
                            _value(
                                row,
                                "CarrierTrackingStatus",
                                "carrierTrackingStatus",
                                "Status",
                                "status",
                            )
                        ),
                        body_style,
                    ),
                ]
            )

        table = Table(
            table_rows,
            repeatRows=1,
            colWidths=[12 * mm, 42 * mm, 58 * mm, 30 * mm, 90 * mm],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#563AD5")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, -1), font),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8C8C8")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F3FF")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)
        if group_index + 1 < len(groups):
            story.append(PageBreak())

    document.build(story)
    return buffer.getvalue()
