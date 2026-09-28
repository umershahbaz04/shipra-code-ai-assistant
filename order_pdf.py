from __future__ import annotations

import re
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
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
    """Create a concise professional order report from verified API data."""
    buffer = BytesIO()
    font = _font_name()
    styles = getSampleStyleSheet()
    brand_style = ParagraphStyle(
        "ShipraBrand", parent=styles["Title"], fontName=font,
        fontSize=20, leading=22, textColor=colors.HexColor("#563AD5"),
    )
    report_style = ParagraphStyle(
        "ShipraReport", parent=styles["Heading2"], fontName=font,
        fontSize=10, leading=12, textColor=colors.HexColor("#696D78"),
    )
    body_style = ParagraphStyle(
        "ShipraBody", parent=styles["BodyText"], fontName=font,
        fontSize=8.5, leading=11, wordWrap="CJK",
    )
    small_style = ParagraphStyle(
        "ShipraSmall", parent=body_style, fontSize=7.2, leading=9,
        textColor=colors.HexColor("#696D78"),
    )
    order_no_style = ParagraphStyle(
        "ShipraOrderNo", parent=styles["Heading1"], fontName=font,
        fontSize=14, leading=17, alignment=TA_RIGHT,
        textColor=colors.HexColor("#22242A"),
    )
    heading_style = ParagraphStyle(
        "ShipraHeading", parent=styles["Heading2"], fontName=font,
        fontSize=10.5, leading=13, textColor=colors.HexColor("#30226F"),
        spaceBefore=7, spaceAfter=4,
    )
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
        title=title, author="Shipra AI Assistant",
    )

    total_orders = sum(len(group.get("rows") or []) for group in groups)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    story: list[Any] = []
    order_number = 0

    for group_index, group in enumerate(groups):
        group_name = str(group.get("name") or "All stores")
        rows = group.get("rows") or []
        for row_index, row in enumerate(rows):
            order_number += 1
            payload = _order_payload(row)
            summary = row.get("summary") if isinstance(row, dict) else {}
            summary = summary if isinstance(summary, dict) else {}
            order = payload.get("order") or payload.get("Order") or payload
            order = order if isinstance(order, dict) else {}
            address = payload.get("orderAddress") or payload.get("OrderAddress") or {}
            address = address if isinstance(address, dict) else {}
            items = payload.get("orderItems") or payload.get("OrderItems") or []
            items = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
            boxes = payload.get("orderBoxes") or payload.get("OrderBoxes") or []
            boxes = [box for box in boxes if isinstance(box, dict)] if isinstance(boxes, list) else []
            reference = _order_reference(payload, order_number)

            header = Table(
                [[
                    [Paragraph("SHIPRA", brand_style), Paragraph("ORDER REPORT", report_style)],
                    Paragraph(f"Order #{escape(reference)}", order_no_style),
                ]],
                colWidths=[90 * mm, 90 * mm],
            )
            header.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 1.2, colors.HexColor("#563AD5")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]))
            story.extend([
                header,
                Spacer(1, 2 * mm),
                Paragraph(
                    escape(f"{filter_text} | Order {order_number} of {total_orders} | Generated {generated_at}"),
                    small_style,
                ),
            ])

            payment_id = _pick(order, "paymentStatusId")
            payment_status = _pick(order, "paymentStatus", "paymentStatusName")
            if not _shown(payment_status):
                payment_status = {1: "Unpaid", 2: "Paid"}.get(payment_id)
            tracking_status = _pick(
                summary, "carrierTrackingStatus", "status",
                default=_pick(order, "carrierTrackingStatus", "status"),
            )
            overview = _info_table([
                ("Order Date", _format_date(_pick(order, "orderDate", default=_pick(summary, "orderDate")))),
                ("Store", group_name),
                ("Order Status", tracking_status),
                ("Payment Status", payment_status),
                ("Items", _pick(order, "itemsCount", default=len(items) or None)),
                ("Sales Channel", _pick(summary, "saleChannelName", "channelName")),
            ], body_style=body_style, font=font)
            if overview:
                story.extend([_section_heading("Order Overview", heading_style), overview])

            city_area = " / ".join(
                str(value) for value in [_named_value(_pick(address, "city")), _named_value(_pick(address, "area"))]
                if _shown(value)
            )
            customer = _info_table([
                ("Customer", _pick(address, "customerName")),
                ("Mobile", _pick(address, "mobile1", "phone")),
                ("Email", _pick(address, "email")),
                ("City / Area", city_area),
                ("Delivery Address", _pick(address, "customerFullAddress", "fullAddress", "streetAddress")),
                ("Country", _named_value(_pick(address, "countryName", "country"))),
            ], body_style=body_style, font=font)
            if customer:
                story.extend([_section_heading("Customer & Delivery", heading_style), customer])

            item_table = _items_table(items, body_style=body_style, font=font)
            if item_table:
                story.extend([_section_heading("Order Items", heading_style), item_table])

            financial = _info_table([
                ("Item Value", _amount(_pick(order, "itemValue"))),
                ("Delivery Charges", _amount(_pick(order, "deliveryCharges"))),
                ("Shipping Charges", _amount(_pick(order, "cShippingCharges", "shippingCharges"))),
                ("Discount", _amount(_pick(order, "discount"))),
                ("VAT / Tax", _amount(_pick(order, "vat", "tax"))),
                ("Total Amount", _amount(_pick(order, "amount", default=_pick(summary, "amount")))),
                ("Actual Amount", _amount(_pick(order, "actualAmount"))),
            ], body_style=body_style, font=font)
            if financial:
                story.extend([_section_heading("Payment Summary", heading_style), financial])

            dimensions = []
            for box in boxes:
                length, width, height = (
                    _pick(box, "length"), _pick(box, "width"), _pick(box, "height")
                )
                if all(_shown(value) for value in (length, width, height)):
                    dimensions.append(f"{length} x {width} x {height}")
            delivery = _info_table([
                ("Tracking Status", tracking_status),
                ("Tracking Number", _pick(summary, "carrierTrackingNo", "trackingNo", default=_pick(order, "carrierTrackingNo", "trackingNo"))),
                ("Carrier", _pick(summary, "carrierName", default=_pick(order, "carrierName"))),
                ("Weight", _pick(order, "weight")),
                ("Package Size", "; ".join(dimensions)),
                ("Packages", len(boxes) if boxes else None),
            ], body_style=body_style, font=font)
            if delivery:
                story.extend([_section_heading("Delivery & Package", heading_style), delivery])

            note_data = payload.get("orderNote") or payload.get("OrderNote") or {}
            note = _pick(note_data, "note") if isinstance(note_data, dict) else note_data
            notes = _info_table([
                ("Description", _pick(order, "description")),
                ("Remarks", _pick(order, "remarks")),
                ("Order Note", note),
            ], body_style=body_style, font=font)
            if notes:
                story.extend([_section_heading("Notes", heading_style), notes])

            is_last_order = group_index == len(groups) - 1 and row_index == len(rows) - 1
            if not is_last_order:
                story.append(PageBreak())

    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 7)
        canvas.setFillColor(colors.HexColor("#777A83"))
        canvas.drawString(15 * mm, 8 * mm, "Shipra - Confidential Order Report")
        canvas.drawRightString(195 * mm, 8 * mm, f"Page {doc.page}")
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
