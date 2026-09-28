from __future__ import annotations

import re
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
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


def _pick(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    if not isinstance(data, dict):
        return default
    lowered = {str(key).lower(): value for key, value in data.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, ""):
            return value
    return default


def _shown(value: Any) -> bool:
    return value not in (None, "", [], {})


def _paragraph(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(value)).replace("\n", "<br/>"), style)


def _format_date(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return value.replace("T", " ").split(".", 1)[0]


def _named_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return None
    text = str(value or "").strip()
    if re.fullmatch(r"\d+(?:_\d+)?", text):
        return None
    return text or None


def _amount(value: Any) -> Any:
    if not _shown(value):
        return None
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return value


def _section_heading(title: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(title), style)


def _info_table(
    pairs: list[tuple[str, Any]],
    *,
    body_style: ParagraphStyle,
    font: str,
) -> Table | None:
    pairs = [(label, value) for label, value in pairs if _shown(value)]
    if not pairs:
        return None
    rows: list[list[Any]] = []
    for offset in range(0, len(pairs), 2):
        row: list[Any] = []
        for label, value in pairs[offset : offset + 2]:
            row.extend(
                [
                    Paragraph(f"<b>{escape(label)}</b>", body_style),
                    _paragraph(value, body_style),
                ]
            )
        while len(row) < 4:
            row.extend(["", ""])
        rows.append(row)
    table = Table(rows, colWidths=[29 * mm, 61 * mm, 29 * mm, 61 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DADDE5")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F1FF")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F3F1FF")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _items_table(
    items: list[dict[str, Any]],
    *,
    body_style: ParagraphStyle,
    font: str,
) -> Table | None:
    if not items:
        return None
    rows: list[list[Any]] = [["#", "Product / SKU", "Qty", "Unit Price", "Discount"]]
    for index, item in enumerate(items, 1):
        name = _pick(item, "productName", "name", default=f"Item {index}")
        sku = _pick(item, "stockSku", "sku")
        product = str(name)
        if _shown(sku):
            product += f"\nSKU: {sku}"
        rows.append(
            [
                str(index),
                _paragraph(product, body_style),
                _paragraph(_pick(item, "quantity", default="-"), body_style),
                _paragraph(_amount(_pick(item, "price", "unitRate")) or "-", body_style),
                _paragraph(_amount(_pick(item, "discount")) or "-", body_style),
            ]
        )
    table = Table(
        rows,
        repeatRows=1,
        colWidths=[10 * mm, 80 * mm, 20 * mm, 35 * mm, 35 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#563AD5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DADDE5")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFD")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


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
    """Create a compact Shopify-style table with one order per row."""
    buffer = BytesIO()
    font = _font_name()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ShipraListTitle", parent=styles["Title"], fontName=font,
        fontSize=18, leading=22, textColor=colors.HexColor("#22242A"),
        alignment=TA_CENTER, spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "ShipraListBody", parent=styles["BodyText"], fontName=font,
        fontSize=6.5, leading=8, wordWrap="CJK",
    )
    meta_style = ParagraphStyle(
        "ShipraListMeta", parent=body_style, fontSize=7.5, leading=10,
        textColor=colors.HexColor("#696D78"),
    )
    document = SimpleDocTemplate(
        buffer, pagesize=landscape(A4), rightMargin=10 * mm, leftMargin=10 * mm,
        topMargin=11 * mm, bottomMargin=12 * mm,
        title=title, author="Shipra AI Assistant",
    )

    total_orders = sum(len(group.get("rows") or []) for group in groups)
    story: list[Any] = [
        Paragraph(escape(title), title_style),
        Paragraph(escape(filter_text), meta_style),
        Paragraph(
            escape(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Total orders: {total_orders}"),
            meta_style,
        ),
        Spacer(1, 4 * mm),
    ]
    table_rows: list[list[Any]] = [[
        "Order", "Date", "Customer", "Store / Channel", "Total",
        "Payment", "Order Status", "Items", "Delivery / Carrier", "Tracking No",
    ]]

    order_number = 0
    for group in groups:
        group_name = str(group.get("name") or "All stores")
        for row in group.get("rows") or []:
            order_number += 1
            payload = _order_payload(row)
            summary = row.get("summary") if isinstance(row, dict) else {}
            summary = summary if isinstance(summary, dict) else {}
            order = payload.get("order") or payload.get("Order") or payload
            order = order if isinstance(order, dict) else {}
            address = payload.get("orderAddress") or payload.get("OrderAddress") or {}
            address = address if isinstance(address, dict) else {}
            items = payload.get("orderItems") or payload.get("OrderItems") or []
            items = items if isinstance(items, list) else []

            payment_id = _pick(order, "paymentStatusId")
            payment = _pick(summary, "paymentStatus", default=_pick(order, "paymentStatus", "paymentStatusName"))
            if not _shown(payment):
                payment = {1: "Unpaid", 2: "Paid"}.get(payment_id, "-")
            order_status = _pick(summary, "orderStatus", "status", default=_pick(order, "orderStatus", "status"))
            delivery_status = _pick(summary, "carrierTrackingStatus", "deliveryStatus", default=_pick(order, "carrierTrackingStatus"))
            carrier = _pick(summary, "carrierName", "deliveryMethod", default=_pick(order, "carrierName"))
            delivery = " / ".join(str(value) for value in (delivery_status, carrier) if _shown(value)) or "-"
            source = _pick(summary, "storeName", "saleChannelName", "channelName")
            if not _shown(source):
                source = group_name

            values = [
                f"#{_order_reference(payload, order_number)}",
                _format_date(_pick(order, "orderDate", default=_pick(summary, "orderDate"))) or "-",
                _pick(address, "customerName", default=_pick(summary, "customerName")) or "-",
                source,
                _amount(_pick(order, "amount", default=_pick(summary, "amount"))) or "-",
                payment,
                order_status or delivery_status or "-",
                _pick(order, "itemsCount", default=len(items)) if items else _pick(order, "itemsCount", default="-"),
                delivery,
                _pick(summary, "carrierTrackingNo", "trackingNo", default=_pick(order, "carrierTrackingNo", "trackingNo")) or "-",
            ]
            table_rows.append([_paragraph(value, body_style) for value in values])

    table = Table(
        table_rows, repeatRows=1,
        colWidths=[23 * mm, 30 * mm, 33 * mm, 31 * mm, 20 * mm,
                   23 * mm, 29 * mm, 13 * mm, 38 * mm, 28 * mm],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#563AD5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D8DAE1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (4, 1), (4, -1), "RIGHT"),
        ("ALIGN", (7, 1), (7, -1), "CENTER"),
    ]))
    story.append(table)

    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 7)
        canvas.setFillColor(colors.HexColor("#777A83"))
        canvas.drawString(10 * mm, 7 * mm, "Shipra - Confidential Order Report")
        canvas.drawRightString(287 * mm, 7 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
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
