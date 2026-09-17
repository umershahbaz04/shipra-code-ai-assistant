from __future__ import annotations

import json
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

    def count_orders(self, kind: str, from_date: date | None, to_date: date | None) -> tuple[int | float, dict[str, str]]:
        route = self.COUNT_ROUTES[kind]
        body = {"filterModel": {
            "createdFrom": from_date.isoformat() if from_date else None,
            "createdTo": to_date.isoformat() if to_date else None,
        }}
        payload = self._request("POST", route, json_body=body)
        count = _find_count(payload)
        if count is None:
            raise ShipraAPIError("The count API returned no recognizable count field.")
        return count, self.auth.as_dict()

    def list_orders(
        self,
        from_date: date | None,
        to_date: date | None,
        search: str = "",
        limit: int = 50,
        start: int = 0,
        carrier_tracking_status_ids: str | None = None,
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
        fetch_limit: int = 50,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Return one consistently filtered count/list result, paging when requested."""
        fetch_limit = min(max(fetch_limit, 1), 1000)
        page_size = min(fetch_limit, 100)
        rows: list[dict[str, Any]] = []
        total: int | float | None = None
        start = 0
        while len(rows) < fetch_limit:
            page, _ = self.list_orders(
                from_date,
                to_date,
                limit=min(page_size, fetch_limit - len(rows)),
                start=start,
                carrier_tracking_status_ids=carrier_tracking_status_ids,
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
        "orderNo", "OrderNo", "orderId", "OrderId", "orderDate", "OrderDate",
        "customerName", "CustomerName", "fullName", "FullName", "amount", "Amount",
        "status", "Status", "carrierTrackingStatus", "CarrierTrackingStatus",
        "carrierTrackingNo", "CarrierTrackingNo",
    ]
    total = count if count is not None else len(rows)
    if language == "Arabic":
        lines = [f"أعادت واجهة Shipra المباشرة **{total}** من {label}. يتم عرض {min(len(rows), limit)}:"]
    elif language == "Roman Urdu":
        lines = [f"Live Shipra API ne **{total} {label}** return kiye. {min(len(rows), limit)} dikhaye ja rahe hain:"]
    else:
        lines = [f"Live Shipra API returned **{total} {label}**. Showing {min(len(rows), limit)}:"]
    for index, row in enumerate(rows[:limit], 1):
        selected = {key: row[key] for key in preferred if key in row and row[key] not in (None, "")}
        if not selected:
            selected = dict(list(row.items())[:8])
        summary = ", ".join(f"{key}: {value}" for key, value in selected.items())
        lines.append(f"{index}. {summary}")
    return "\n".join(lines)


def format_order_detail(payload: dict[str, Any]) -> str:
    result = payload.get("result", payload)
    return "Live order details:\n\n```json\n" + json.dumps(result, indent=2, ensure_ascii=False, default=str)[:12000] + "\n```"
