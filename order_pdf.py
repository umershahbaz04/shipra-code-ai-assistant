from __future__ import annotations

import re
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

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


def _flatten_details(
    value: Any,
    path: str = "",
) -> Iterator[tuple[str, str]]:
    """Yield every value from a nested Shipra order response."""
    if isinstance(value, dict):
        if not value:
            yield path or "value", "{}"
            return
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _flatten_details(child, child_path)
        return

    if isinstance(value, list):
        if not value:
            yield path or "value", "[]"
            return
        for index, child in enumerate(value, 1):
            child_path = f"{path}[{index}]" if path else f"item[{index}]"
            yield from _flatten_details(child, child_path)
        return

    if value is None:
        text = "null"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    yield path or "value", text


def _order_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("details", row)
    if not isinstance(payload, dict):
        return {"value": payload}
    result = payload.get("result", payload)
    return result if isinstance(result, dict) else {"result": result}


def _order_reference(payload: dict[str, Any], fallback: int) -> str:
    order = payload.get("order") or payload.get("Order") or payload
    if isinstance(order, dict):
        return _value(
            order,
            "orderNo",
            "OrderNo",
            "refNo",
            "RefNo",
            "orderId",
            "OrderId",
            default=f"Order {fallback}",
        )
    return f"Order {fallback}"


def build_orders_pdf(
    groups: list[dict[str, Any]],
    *,
    title: str,
    filter_text: str,
) -> bytes:
    """Create a PDF containing every field returned by each order-detail API."""
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
        wordWrap="CJK",
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

    order_number = 0
    for group_index, group in enumerate(groups):
        group_name = str(group.get("name") or "Orders")
        rows = group.get("rows") or []
        story.append(
            Paragraph(
                escape(f"Store: {group_name} ({len(rows)} orders)"),
                heading_style,
            )
        )

        for row_index, row in enumerate(rows):
            order_number += 1
            payload = _order_payload(row)
            reference = _order_reference(payload, order_number)
            story.append(
                Paragraph(
                    escape(f"Order {order_number}: {reference}"),
                    heading_style,
                )
            )
            table_rows: list[list[Any]] = [["Field", "Value"]]
            for field, value in _flatten_details(payload):
                table_rows.append(
                    [
                        Paragraph(escape(field), body_style),
                        Paragraph(escape(value).replace("\n", "<br/>"), body_style),
                    ]
                )

            table = Table(
                table_rows,
                repeatRows=1,
                colWidths=[72 * mm, 192 * mm],
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#563AD5")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, -1), font),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8C8C8")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F3FF")]),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(table)
            is_last_order = (
                group_index == len(groups) - 1
                and row_index == len(rows) - 1
            )
            if not is_last_order:
                story.append(PageBreak())

    document.build(story)
    return buffer.getvalue()


def build_summary_pdf(*, title: str, summary: str) -> bytes:
    """Create a PDF containing the exact privacy-safe order summary shown in chat."""
    buffer = BytesIO()
    font = _font_name()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ShipraSummaryTitle",
        parent=styles["Title"],
        fontName=font,
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "ShipraSummaryBody",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=10,
        leading=15,
        spaceAfter=5,
    )
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="Shipra AI Assistant",
    )
    story = [Paragraph(escape(title), title_style)]
    for line in summary.splitlines():
        clean = line.strip().replace("**", "")
        if clean:
            story.append(Paragraph(escape(clean), body_style))
        else:
            story.append(Spacer(1, 2 * mm))
    document.build(story)
    return buffer.getvalue()
