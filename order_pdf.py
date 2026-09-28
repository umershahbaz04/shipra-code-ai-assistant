from __future__ import annotations

import json
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


SECTION_NAMES = {
    "order": "Order Information",
    "orderItems": "Order Items",
    "orderAddress": "Customer & Delivery Address",
    "orderAddressAdditional": "Additional Address",
    "orderNote": "Order Note",
    "orderTax": "Taxes",
    "orderBoxes": "Boxes & Dimensions",
    "metafields": "Metafields",
    "settingConfig": "Configured Fields",
}


def _is_empty(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, dict):
        return not value or all(_is_empty(child) for child in value.values())
    if isinstance(value, list):
        return not value or all(_is_empty(child) for child in value)
    return False


def _label(name: str) -> str:
    if name in SECTION_NAMES:
        return SECTION_NAMES[name]
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    text = text.replace("_", " ").strip()
    text = re.sub(r"\bId\b", "ID", text, flags=re.IGNORECASE)
    text = re.sub(r"\bUrl\b", "URL", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSku\b", "SKU", text, flags=re.IGNORECASE)
    return text[:1].upper() + text[1:]


def _decoded(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[:1] not in "[{":
        return value
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return value


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _details_table(
    pairs: list[tuple[str, Any]],
    *,
    body_style: ParagraphStyle,
    font: str,
) -> Table:
    rows: list[list[Any]] = []
    for offset in range(0, len(pairs), 2):
        row: list[Any] = []
        for name, value in pairs[offset : offset + 2]:
            row.extend(
                [
                    Paragraph(f"<b>{escape(_label(name))}</b>", body_style),
                    Paragraph(
                        escape(_text(value)).replace("\n", "<br/>"),
                        body_style,
                    ),
                ]
            )
        while len(row) < 4:
            row.extend(["", ""])
        rows.append(row)

    table = Table(
        rows,
        colWidths=[38 * mm, 94 * mm, 38 * mm, 94 * mm],
        splitByRow=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D5D5D5")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F0FF")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F2F0FF")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _append_section(
    story: list[Any],
    name: str,
    value: Any,
    *,
    body_style: ParagraphStyle,
    heading_style: ParagraphStyle,
    font: str,
) -> None:
    value = _decoded(value)
    if _is_empty(value):
        return

    title = _label(name)
    if isinstance(value, dict):
        scalar_pairs: list[tuple[str, Any]] = []
        nested: list[tuple[str, Any]] = []
        for key, child in value.items():
            child = _decoded(child)
            if _is_empty(child):
                continue
            if isinstance(child, (dict, list)):
                nested.append((key, child))
            else:
                scalar_pairs.append((key, child))

        if not scalar_pairs and not nested:
            return
        story.append(Paragraph(escape(title), heading_style))
        if scalar_pairs:
            story.append(
                _details_table(
                    scalar_pairs,
                    body_style=body_style,
                    font=font,
                )
            )
        for key, child in nested:
            _append_section(
                story,
                key,
                child,
                body_style=body_style,
                heading_style=heading_style,
                font=font,
            )
        return

    if isinstance(value, list):
        visible = []
        for item in value:
            item = _decoded(item)
            if not _is_empty(item):
                visible.append(item)
        if not visible:
            return
        story.append(Paragraph(escape(title), heading_style))
        for index, item in enumerate(visible, 1):
            item_title = f"{title[:-1] if title.endswith('s') else title} {index}"
            if isinstance(item, dict):
                pairs: list[tuple[str, Any]] = []
                nested: list[tuple[str, Any]] = []
                for key, child in item.items():
                    child = _decoded(child)
                    if _is_empty(child):
                        continue
                    if isinstance(child, (dict, list)):
                        nested.append((key, child))
                    else:
                        pairs.append((key, child))
                if len(visible) > 1:
                    story.append(
                        Paragraph(
                            f"<b>{escape(item_title)}</b>",
                            body_style,
                        )
                    )
                if pairs:
                    story.append(
                        _details_table(
                            pairs,
                            body_style=body_style,
                            font=font,
                        )
                    )
                for key, child in nested:
                    _append_section(
                        story,
                        key,
                        child,
                        body_style=body_style,
                        heading_style=heading_style,
                        font=font,
                    )
            else:
                story.append(Paragraph(escape(_text(item)), body_style))
        return

    story.append(
        Paragraph(
            f"<b>{escape(title)}:</b> {escape(_text(value))}",
            body_style,
        )
    )


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
            preferred_sections = (
                "order",
                "orderItems",
                "orderAddress",
                "orderNote",
                "orderTax",
                "orderBoxes",
                "metafields",
            )
            used = set()
            for section in preferred_sections:
                if section in payload:
                    used.add(section)
                    _append_section(
                        story,
                        section,
                        payload[section],
                        body_style=body_style,
                        heading_style=heading_style,
                        font=font,
                    )
            for section, value in payload.items():
                if section in used:
                    continue
                _append_section(
                    story,
                    section,
                    value,
                    body_style=body_style,
                    heading_style=heading_style,
                    font=font,
                )
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
