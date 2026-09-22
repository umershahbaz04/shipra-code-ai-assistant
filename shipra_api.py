from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import requests


class ShipraAPIError(RuntimeError):
    pass


def _error_message(payload: Any, fallback: str) -> str:
    if not isinstance(payload, dict):
        return fallback
    errors = payload.get("errors")
    if isinstance(errors, dict):
        messages: list[str] = []
        for value in errors.values():
            if isinstance(value, list):
                messages.extend(str(item) for item in value)
            elif value:
                messages.append(str(value))
        if messages:
            return "; ".join(messages)
    return str(payload.get("message") or fallback)


def _find_count(payload: Any) -> int | float | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.lower() in {"count", "totalcount", "total"} and isinstance(value, (int, float)):
                return value
        for key in ("result", "data"):
            value = payload.get(key)
            found = _find_count(value)
            if found is not None:
                return found
    return None


def _find_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("list", "items", "rows", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        result = payload.get("result")
        rows = _find_rows(result)
        if rows:
            return rows
    return []


@dataclass
class ShipraAuth:
    username: str
    access_token: str
    refresh_token: str
    id_token: str = ""

    @classmethod
    def from_result(cls, username: str, result: dict[str, Any]) -> "ShipraAuth":
        return cls(
            username=str(result.get("user_name") or result.get("userName") or username),
            access_token=str(result.get("access_token") or result.get("accessToken") or ""),
            refresh_token=str(result.get("refresh_token") or result.get("refreshToken") or ""),
            id_token=str(result.get("id_token") or result.get("idToken") or ""),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "username": self.username,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "id_token": self.id_token,
        }


class ShipraAPI:
    """Read-only client for the authenticated Shipra API endpoints."""

    COUNT_ROUTES = {
        "total": "Dashboard/GetTotalNoofOrdersPlacedCounts",
        "delivered": "Dashboard/GetDelieveredOrderCount",
        "in_progress": "Dashboard/GetInProgressOrderCount",
        "returned": "Dashboard/GetReturnedOrderCount",
        "regular": "Dashboard/GetRegularOrderCount",
        "fulfillable": "Dashboard/GetFulFillableOrderCount",
        "to_be_packed": "Dashboard/GetToBePackedCount",
        "to_be_shipped": "Dashboard/GetToBeShippedCount",
    }

    def __init__(self, base_url: str, auth: dict[str, str] | None = None, timeout: int = 20):
        self.base_url = base_url.rstrip("/") + "/"
        self.auth = ShipraAuth(**auth) if auth else None
        self.timeout = timeout

    def _url(self, route: str) -> str:
        return self.base_url + route.lstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth:
            headers["Authorization"] = f"Bearer {self.auth.access_token}"
            headers["x-access-token"] = self.auth.access_token
            if self.auth.id_token:
                headers["x-id-token"] = self.auth.id_token
        return headers

    def login(self, username: str, password: str) -> dict[str, str]:
        response = requests.post(
            self._url("UserManagement/Login"),
            json={"userName": username, "password": password},
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        payload = self._decode(response)
        if response.status_code >= 400 or not payload.get("isSuccess"):
            raise ShipraAPIError(_error_message(payload, "Shipra login failed."))
        result = payload.get("result") or {}
        auth = ShipraAuth.from_result(username, result)
        if not auth.access_token:
            raise ShipraAPIError("Login succeeded but no access token was returned.")
        self.auth = auth
        return auth.as_dict()

    def refresh(self) -> dict[str, str]:
        if not self.auth or not self.auth.refresh_token:
            raise ShipraAPIError("Your Shipra session expired. Please connect again.")
        response = requests.post(
            self._url("UserManagement/GetAccessTokenWithRefreshToken"),
            json={"userName": self.auth.username, "refreshToken": self.auth.refresh_token},
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        payload = self._decode(response)
        if response.status_code >= 400 or not payload.get("isSuccess"):
            raise ShipraAPIError(_error_message(payload, "Unable to refresh the Shipra session."))
        updated = ShipraAuth.from_result(self.auth.username, payload.get("result") or {})
        if not updated.refresh_token:
            updated.refresh_token = self.auth.refresh_token
        if not updated.id_token:
            updated.id_token = self.auth.id_token
        if not updated.access_token:
            raise ShipraAPIError("Token refresh returned no access token.")
        self.auth = updated
        return updated.as_dict()

    @staticmethod
    def _decode(response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ShipraAPIError(f"Shipra API returned an invalid response ({response.status_code}).") from exc
        if not isinstance(payload, dict):
            raise ShipraAPIError("Shipra API returned an unexpected response format.")
        return payload

    def _request(self, method: str, route: str, *, json_body: dict[str, Any] | None = None, retry: bool = True) -> dict[str, Any]:
        if not self.auth:
            raise ShipraAPIError("Connect your Shipra account first.")
        response = requests.request(
            method,
            self._url(route),
            json=json_body,
            headers=self._headers(),
            timeout=self.timeout,
        )
        if response.status_code in {401, 419} and retry:
            self.refresh()
            return self._request(method, route, json_body=json_body, retry=False)
        payload = self._decode(response)
        if response.status_code >= 400 or payload.get("isSuccess") is False:
            raise ShipraAPIError(_error_message(payload, f"Shipra API request failed ({response.status_code})."))
        return payload

    def count_orders(
        self,
        kind: str,
        from_date: date | None,
        to_date: date | None,
    ) -> tuple[int | float, dict[str, str]]:
        route = self.COUNT_ROUTES[kind]

        body = {
            "filterModel": {
                "createdFrom": (
                    from_date.isoformat()
                    if from_date
                    else None
                ),
                "createdTo": (
                    to_date.isoformat()
                    if to_date
                    else None
                ),
            }
        }

        payload = self._request(
            "POST",
            route,
            json_body=body,
        )

        count = _find_count(payload)

        if count is None:
            raise ShipraAPIError(
                "The count API returned no recognizable count field."
            )

        return count, self.auth.as_dict()

    def count_stores(
        self,
    ) -> tuple[int, dict[str, str]]:
        body = {
            "filterModel": {
                "createdFrom": None,
                "createdTo": None,
                "start": 0,
                "length": 1,
                "search": "",
                "sortCol": 0,
                "sortDir": "desc",
            }
        }

        payload = self._request(
            "POST",
            "Store/GetAllStores",
            json_body=body,
        )

        count = _find_count(payload)

        if count is None:
            raise ShipraAPIError(
                "Store API returned no recognizable TotalCount."
            )

        return int(count), self.auth.as_dict()
    
    def list_all_stores(
        self,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, str],
    ]:
        stores: list[dict[str, Any]] = []
        seen_store_ids: set[int] = set()
        start = 0
        page_size = 100
        total_count: int | None = None

        while total_count is None or start < total_count:
            body = {
                "filterModel": {
                    "createdFrom": None,
                    "createdTo": None,
                    "start": start,
                    "length": page_size,
                    "search": "",
                    "sortCol": 0,
                    "sortDir": "desc",
                }
            }

            payload = self._request(
                "POST",
                "Store/GetAllStores",
                json_body=body,
            )

            rows = _find_rows(payload)
            found_count = _find_count(payload)

            if total_count is None:
                total_count = int(
                    found_count
                    if found_count is not None
                    else len(rows)
                )

            for store in rows:
                store_id = (
                    store.get("StoreId")
                    or store.get("storeId")
                )

                if store_id is None:
                    continue

                store_id = int(store_id)

                if store_id in seen_store_ids:
                    continue

                seen_store_ids.add(store_id)
                stores.append(store)

            if not rows:
                break

            start += len(rows)

        return stores, self.auth.as_dict()

    def count_orders_by_store(
        self,
        from_date: date | None,
        to_date: date | None,
        carrier_tracking_status_ids: str | None = None,
        payment_status_id: int | None = None,
    ) -> tuple[
        dict[str, Any],
        dict[str, str],
    ]:
        overall_data, _ = self.list_orders(
            from_date=from_date,
            to_date=to_date,
            limit=1,
            start=0,
            carrier_tracking_status_ids=(
                carrier_tracking_status_ids
            ),
            payment_status_id=payment_status_id,
        )

        overall_total = int(
            overall_data.get("count") or 0
        )

        stores, _ = self.list_all_stores()

        store_counts: list[dict[str, Any]] = []
        grouped_total = 0

        for store in stores:
            store_id = (
                store.get("StoreId")
                or store.get("storeId")
            )

            store_name = (
                store.get("StoreName")
                or store.get("storeName")
                or f"Store {store_id}"
            )

            if store_id is None:
                continue

            store_data, _ = self.list_orders(
                from_date=from_date,
                to_date=to_date,
                limit=1,
                start=0,
                carrier_tracking_status_ids=(
                    carrier_tracking_status_ids
                ),
                payment_status_id=payment_status_id,
                store_ids=str(store_id),
            )

            order_count = int(
                store_data.get("count") or 0
            )

            grouped_total += order_count

            store_counts.append(
                {
                    "storeId": int(store_id),
                    "storeName": str(store_name),
                    "orderCount": order_count,
                }
            )

        store_counts.sort(
            key=lambda item: item["orderCount"],
            reverse=True,
        )

        if grouped_total != overall_total:
            raise ShipraAPIError(
                "Store-wise order total does not "
                "match the overall filtered total. "
                "No result was shown."
            )

        result = {
            "totalCount": overall_total,
            "groupedTotal": grouped_total,
            "stores": store_counts,
        }

        return result, self.auth.as_dict()
    
    def list_orders(
        self,
        from_date: date | None,
        to_date: date | None,
        search: str = "",
        limit: int = 50,
        start: int = 0,
        carrier_tracking_status_ids: str | None = None,
        payment_status_id: int | None = None,
        store_ids: str | None = None,        
    ) -> tuple[dict[str, Any], dict[str, str]]:
        body = {
            "filterModel": {
                "createdFrom": from_date.isoformat() if from_date else None,
                "createdTo": to_date.isoformat() if to_date else None,
                "start": max(start, 0),
                "length": min(max(limit, 1), 100),
                "search": search,
                "sortDir": "desc",
                "sortCol": 0,
            },
            "orderRequestVia": 0,
            "readyForAssignment": True,
            "carrierTrackingStatusIds": carrier_tracking_status_ids,
            "paymentStatusId": payment_status_id,
            "storeIds": store_ids,            
            "orderAddressFilter": {},
        }
        payload = self._request("POST", "Order/GetAllOrders", json_body=body)
        return {"count": _find_count(payload), "rows": _find_rows(payload), "raw": payload}, self.auth.as_dict()

    def search_orders(
        self,
        from_date: date | None,
        to_date: date | None,
        *,
        carrier_tracking_status_ids: str | None = None,
        payment_status_id: int | None = None,
        fetch_limit: int = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Return one consistently filtered count/list result, paging when requested."""

        page_size = min(fetch_limit, 100)
        rows: list[dict[str, Any]] = []
        total: int | float | None = None
        start = 0
        while total is None or len(rows) < total:
            page, _ = self.list_orders(
                from_date,
                to_date,
                limit=min(page_size, fetch_limit - len(rows)),
                start=start,
                carrier_tracking_status_ids=carrier_tracking_status_ids,
                payment_status_id=payment_status_id,
            )
            page_rows = page.get("rows") or []
            page_total = page.get("count")
            if total is None:
                total = page_total if page_total is not None else len(page_rows)
            elif page_total is not None and page_total != total:
                raise ShipraAPIError("Order total changed while paging; please retry for a consistent result.")
            rows.extend(page_rows)
            start += len(page_rows)
            if not page_rows or start >= int(total or 0):
                break
        return {"count": total or 0, "rows": rows, "complete": len(rows) >= int(total or 0)}, self.auth.as_dict()

    def order_by_id(self, order_id: str) -> tuple[dict[str, Any], dict[str, str]]:
        payload = self._request("GET", f"Order/GetOrderById?OrderId={order_id}")
        return payload, self.auth.as_dict()

    def order_by_reference(self, reference: str) -> tuple[dict[str, Any], dict[str, str]]:
        """Resolve an order number/reference through the read-only list API first."""
        today = date.today()
        data, _ = self.list_orders(today - timedelta(days=365 * 5), today, search=reference, limit=5)
        rows = data.get("rows") or []
        if not rows:
            raise ShipraAPIError(f"No order matched reference {reference}.")
        row = rows[0]
        order_id = row.get("orderId") or row.get("OrderId")
        if order_id:
            return self.order_by_id(str(order_id))
        return {"isSuccess": True, "result": row}, self.auth.as_dict()


def live_order_intent(question: str, *, order_context: bool = False) -> dict[str, str] | None:
    text = question.lower().strip()
    # Normalize common typing variations before deterministic intent matching.
    text = re.sub(r"\b(?:penidng|pening|pendng|panding)\b", "pending", text)
    text = re.sub(r"\b(?:orde|ordr|oders)\b", "order", text)
    has_order_word = bool(re.search(r"\b(order|orders|orderon|آرڈر|طلب|طلبات|الطلبات)\b", text))
    if not has_order_word and not order_context:
        return None
    live_words = r"\b(today|aaj|aj|yesterday|kal|count|kitn(?:a|e|i|y)|how many|show|list|details?|status|deliver(?:ed)?|returned|pending|packed|shipped|tracking|total|kul|last|past|akhri|pichl[aeiy]*)\b|كم|عدد|إجمالي"
    if not re.search(live_words, text):
        return None

    guid = re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", text, re.IGNORECASE)
    if guid:
        return {"action": "detail", "order_id": guid.group(0)}

    if re.search(r"\b(details?|status|tracking)\b", text):
        reference = re.search(
            r"\border(?:\s*(?:no|number|#|id))?\s*[:#-]?\s*([a-z0-9][a-z0-9_-]{2,})\b",
            text,
            re.IGNORECASE,
        )
        if reference and reference.group(1) not in {"detail", "details", "status", "tracking", "today"}:
            return {"action": "detail_search", "reference": reference.group(1)}

    count_kind = "total"
    if re.search(r"\b(delivered|deliver(?:ed)?|pohnch|pahunch)\b", text):
        count_kind = "delivered"
    elif re.search(r"\b(in[ -]?progress|processing|pending)\b", text):
        count_kind = "in_progress"
    elif re.search(r"\b(returned|return)\b", text):
        count_kind = "returned"
    elif re.search(r"\bregular\b", text):
        count_kind = "regular"
    elif re.search(r"\bfulfill(?:able|ment)?\b", text):
        count_kind = "fulfillable"
    elif re.search(r"\b(pack|packed|packing)\b", text):
        count_kind = "to_be_packed"
    elif re.search(r"\b(ship|shipped|shipping)\b", text):
        count_kind = "to_be_shipped"

    if re.search(r"\b(count|how many|kitn(?:a|e|i|y)|total|kul)\b|كم|عدد|إجمالي", text):
        return {"action": "count", "kind": count_kind}
    return {"action": "list", "kind": count_kind}


def filter_in_progress_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the same exclusions used by DashboardRepository.GetInProgressOrderCount."""
    terminal_ids = {6, 26}  # Delivered, ReturnToOrigin
    terminal_names = {"delivered", "returntoorigin", "return to origin"}
    filtered: list[dict[str, Any]] = []
    for row in rows:
        raw_id = row.get("carrierTrackingStatusId", row.get("CarrierTrackingStatusId"))
        raw_name = row.get("carrierTrackingStatus", row.get("CarrierTrackingStatus", ""))
        try:
            status_id = int(raw_id) if raw_id is not None else None
        except (TypeError, ValueError):
            status_id = None
        status_name = re.sub(r"[_-]+", " ", str(raw_name)).strip().lower()
        compact_name = status_name.replace(" ", "")
        if status_id in terminal_ids or status_name in terminal_names or compact_name in terminal_names:
            continue
        filtered.append(row)
    return filtered


def format_order_rows(data: dict[str, Any], limit: int = 10, language: str = "English", label: str = "orders") -> str:
    rows = data.get("rows") or []
    count = data.get("count")
    if not rows:
        if language == "Arabic":
            return f"لم يتم العثور على طلبات مطابقة. العدد الإجمالي: {count or 0}."
        if language == "Roman Urdu":
            return f"Koi matching order nahi mila. Total count: {count or 0}."
        return f"No matching orders were returned. Total count: {count or 0}."

    preferred = [
        "orderNo", "OrderNo", "orderDate", "OrderDate", "amount", "Amount",
        "status", "Status", "carrierTrackingStatus", "CarrierTrackingStatus",
    ]
    total = count if count is not None else len(rows)
    if language == "Arabic":
        lines = [f"أعادت واجهة Shipra المباشرة **{total}** من {label}. يتم عرض {min(len(rows), limit)}:"]
    elif language == "Roman Urdu":
        lines = [f"Live Shipra API ne **{total} {label}** return kiye. {len(rows)} dikhaye ja rahe hain:"]
    else:
        lines = [f"Live Shipra API returned **{total} {label}**. Showing {len(rows)}:"]
    for index, row in enumerate(rows, 1):
        selected = {key: row[key] for key in preferred if key in row and row[key] not in (None, "")}
        summary = ", ".join(f"{key}: {value}" for key, value in selected.items()) or "No safe summary fields returned"
        lines.append(f"{index}. {summary}")
    return "\n".join(lines)


def _mask_name(value: Any) -> str:
    words = str(value or "").strip().split()
    return " ".join(word[:1] + "*" * max(len(word) - 1, 2) for word in words) or "-"


def _mask_phone(value: Any) -> str:
    text = str(value or "").strip()
    return ("*" * max(len(text) - 4, 4) + text[-4:]) if text else "-"


def _mask_email(value: Any) -> str:
    text = str(value or "").strip()
    if "@" not in text:
        return "-"
    local, domain = text.split("@", 1)
    return (local[:1] or "*") + "***@" + domain


def format_order_detail(payload: dict[str, Any], language: str = "English") -> str:
    """Render a privacy-safe order summary instead of exposing the raw API object."""
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        raise ShipraAPIError("Shipra returned an unexpected order-detail format.")
    order = result.get("order") or result.get("Order") or result
    address = result.get("orderAddress") or result.get("OrderAddress") or {}
    items = result.get("orderItems") or result.get("OrderItems") or []
    if not isinstance(order, dict):
        raise ShipraAPIError("Shipra returned no recognizable order object.")

    payment_id = order.get("paymentStatusId", order.get("PaymentStatusId"))
    payment_status = {1: "Unpaid", 2: "Paid"}.get(payment_id, str(payment_id or "-"))
    reference = order.get("orderNo") or order.get("OrderNo") or order.get("refNo") or order.get("RefNo") or "-"
    tracking = order.get("carrierTrackingStatus") or order.get("CarrierTrackingStatus") or "-"

    lines = [
        "Live order summary:",
        f"- Order reference: **{reference}**",
        f"- Order date: {order.get('orderDate', order.get('OrderDate', '-'))}",
        f"- Amount: {order.get('amount', order.get('Amount', '-'))}",
        f"- Payment status: {payment_status}",
        f"- Tracking status: {tracking}",
        f"- Items count: {order.get('itemsCount', order.get('ItemsCount', len(items) if isinstance(items, list) else '-'))}",
        f"- Customer: {_mask_name(address.get('customerName', address.get('CustomerName', '')) if isinstance(address, dict) else '')}",
        f"- Mobile: {_mask_phone(address.get('mobile1', address.get('Mobile1', '')) if isinstance(address, dict) else '')}",
        f"- Email: {_mask_email(address.get('email', address.get('Email', '')) if isinstance(address, dict) else '')}",
    ]
    if isinstance(items, list) and items:
        lines.append("- Items:")
        for item in items[:10]:
            if not isinstance(item, dict):
                continue
            name = item.get("productName") or item.get("ProductName") or "Unnamed item"
            quantity = item.get("quantity", item.get("Quantity", "-"))
            price = item.get("price", item.get("Price", "-"))
            lines.append(f"  - {name} — quantity: {quantity}, price: {price}")
        if len(items) > 10:
            lines.append(f"  - {len(items) - 10} more item(s) hidden from the summary")
    lines.append("\nSensitive address, full contact details, internal IDs, notes, and metadata are hidden by default.")
    return "\n".join(lines)
