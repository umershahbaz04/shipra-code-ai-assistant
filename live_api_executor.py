from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from live_api_intent import LiveAPIIntent
from live_api_registry import get_tool
from shipra_api import ShipraAPI, ShipraAPIError, _find_count, _find_rows


@dataclass(frozen=True)
class LiveAPIResult:
    tool: str
    domain: str
    operation: str
    count: int
    rows: list[dict[str, Any]]
    detail: dict[str, Any] | None
    auth: dict[str, str]


def execute_live_intent(api: ShipraAPI, intent: LiveAPIIntent) -> LiveAPIResult:
    if not intent.is_live_query or not intent.tool:
        raise ValueError("No executable live-data intent was supplied.")
    if intent.needs_clarification:
        raise ValueError("A request requiring clarification cannot be executed.")

    tool = get_tool(intent.tool)
    parameters = dict(intent.parameters or {})
    missing = [key for key in tool.required_parameters if parameters.get(key) in {None, ""}]
    if missing:
        raise ValueError(f"Missing required parameter: {', '.join(missing)}")

    route = tool.route + (tool.query_builder(parameters) if tool.query_builder else "")
    body = tool.body_builder(parameters) if tool.body_builder else None
    payload = api._request(tool.method, route, json_body=body)
    result = payload.get("result")
    rows = _find_rows(payload)
    detail = result if tool.operation == "detail" and isinstance(result, dict) else None
    count_value = _find_count(payload)
    count = int(count_value) if count_value is not None else len(rows)

    if tool.operation == "detail" and detail is None:
        raise ShipraAPIError("Shipra returned no recognizable detail object.")

    return LiveAPIResult(
        tool=tool.name,
        domain=tool.domain,
        operation=tool.operation,
        count=count,
        rows=rows,
        detail=detail,
        auth=api.auth.as_dict(),
    )
