from __future__ import annotations

from typing import Any

from live_api_executor import LiveAPIResult


SENSITIVE_KEYS = {
    "password", "token", "accesstoken", "refreshtoken", "idtoken",
    "connectionstring", "secret", "apikey", "key",
}
DISPLAY_KEYS = (
    "StoreId", "storeId", "StoreName", "storeName", "StoreCode", "storeCode",
    "ProductStationId", "productStationId", "SName", "sname", "Name", "name",
    "Id", "id", "Active", "active", "CountryName", "countryName",
    "SaleChannelName", "saleChannelName", "StatusName", "statusName",
    "OrderTypeName", "orderTypeName", "PMName", "pmName", "FullAddress", "fullAddress",
)


def _safe_fields(row: dict[str, Any]) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    for key in DISPLAY_KEYS:
        if key in row and row[key] is not None and row[key] != "":
            normalized = key.lower().replace("_", "")
            if not any(secret in normalized for secret in SENSITIVE_KEYS):
                values.append((key, row[key]))
        if len(values) >= 6:
            break
    if values:
        return values
    for key, value in row.items():
        normalized = key.lower().replace("_", "")
        if any(secret in normalized for secret in SENSITIVE_KEYS):
            continue
        if isinstance(value, (str, int, float, bool)) and value is not None and value != "":
            values.append((key, value))
        if len(values) >= 6:
            break
    return values


def format_live_result(result: LiveAPIResult, language: str, *, count_only: bool = False) -> str:
    domain = result.domain.replace("_", " ")
    if count_only:
        if language == "Arabic":
            return f"وفقاً لواجهة Shipra المباشرة، العدد هو **{result.count}** ({domain})."
        if language == "Roman Urdu":
            return f"Live Shipra API ke mutabiq **{result.count} {domain}** hain."
        return f"According to the live Shipra API, there are **{result.count} {domain}**."

    source_rows = [result.detail] if result.detail else result.rows
    rows = [row for row in source_rows if isinstance(row, dict)]
    if not rows:
        return "Koi matching live data nahi mila." if language == "Roman Urdu" else "No matching live data was returned."

    intro = (
        f"Live Shipra API ne {result.count} {domain} return kiye:"
        if language == "Roman Urdu"
        else f"Live Shipra API returned {result.count} {domain}:"
    )
    lines = [intro]
    for index, row in enumerate(rows[:50], 1):
        fields = ", ".join(f"{key}: {value}" for key, value in _safe_fields(row))
        lines.append(f"{index}. {fields or 'No safe display fields'}")
    if result.count > len(rows):
        lines.append(f"\nShowing {len(rows)} of {result.count}.")
    return "\n".join(lines)
