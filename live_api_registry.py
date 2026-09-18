from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ReadOnlyTool:
    name: str
    domain: str
    operation: str
    method: str
    route: str
    required_parameters: tuple[str, ...] = ()
    body_builder: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None
    query_builder: Callable[[dict[str, Any]], str] | None = None


def paged_body(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "filterModel": {
            "createdFrom": None,
            "createdTo": None,
            "start": max(int(parameters.get("start", 0)), 0),
            "length": min(max(int(parameters.get("limit", 50)), 1), 100),
            "search": str(parameters.get("search") or ""),
            "sortCol": 0,
            "sortDir": "desc",
        }
    }


def _positive_int(parameters: dict[str, Any], key: str) -> int:
    value = int(parameters[key])
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


READ_ONLY_TOOLS: dict[str, ReadOnlyTool] = {
    "store_list": ReadOnlyTool(
        "store_list", "stores", "list", "POST", "Store/GetAllStores",
        body_builder=paged_body,
    ),
    "store_detail": ReadOnlyTool(
        "store_detail", "stores", "detail", "GET", "Store/GetStoreById",
        required_parameters=("store_id",),
        query_builder=lambda p: f"?StoreId={_positive_int(p, 'store_id')}",
    ),
    "active_store_list": ReadOnlyTool(
        "active_store_list", "stores", "list", "GET", "Store/GetStoresForSelection",
    ),
    "station_list": ReadOnlyTool(
        "station_list", "stations", "list", "GET", "CommonLookup/GetAllStationLookup",
    ),
    "carrier_list": ReadOnlyTool(
        "carrier_list", "carriers", "list", "GET", "Carrier/GetActiveCarriersForSelection",
    ),
    "payment_status_list": ReadOnlyTool(
        "payment_status_list", "payment_statuses", "list", "GET", "CommonLookup/GetAllPaymentStatusLookup",
    ),
    "payment_method_list": ReadOnlyTool(
        "payment_method_list", "payment_methods", "list", "GET", "CommonLookup/GetAllPaymentMethodLookup",
    ),
    "order_type_list": ReadOnlyTool(
        "order_type_list", "order_types", "list", "GET", "CommonLookup/GetAllOrderTypeLookup",
    ),
    "tracking_status_list": ReadOnlyTool(
        "tracking_status_list", "tracking_statuses", "list", "GET", "CommonLookup/GetAllCarrierTrackingStatusLookupForSelection",
    ),
    "product_category_list": ReadOnlyTool(
        "product_category_list", "product_categories", "list", "GET", "ProductCategory/GetAllProductCategoryLookup",
    ),
    "product_detail": ReadOnlyTool(
        "product_detail", "products", "detail", "GET", "Product/GetProductById",
        required_parameters=("product_id",),
        query_builder=lambda p: f"?ProductId={p['product_id']}",
    ),
    "sale_channels_for_store": ReadOnlyTool(
        "sale_channels_for_store", "sale_channels", "list", "GET", "SaleChannel/GetAllSaleChannelsByStoreId",
        required_parameters=("store_id",),
        query_builder=lambda p: f"?StoreId={_positive_int(p, 'store_id')}",
    ),
    "sale_channel_detail": ReadOnlyTool(
        "sale_channel_detail", "sale_channels", "detail", "GET", "SaleChannel/GetSaleChannelConfigById",
        required_parameters=("sale_channel_config_id",),
        query_builder=lambda p: (
            f"?SaleChannelConfigId={_positive_int(p, 'sale_channel_config_id')}"
        ),
    ),
}


BLOCKED_ROUTE_WORDS = {
    "create", "update", "delete", "disable", "enable", "activate",
    "upload", "assign", "generate", "sync", "wipe", "rotate", "mark",
}


def get_tool(name: str) -> ReadOnlyTool:
    tool = READ_ONLY_TOOLS.get(name)
    if tool is None:
        raise ValueError("The requested live-data tool is not allow-listed.")
    route_words = tool.route.lower()
    if any(word in route_words for word in BLOCKED_ROUTE_WORDS):
        raise ValueError("A mutation-like route was blocked by the safety policy.")
    if tool.method not in {"GET", "POST"}:
        raise ValueError("Unsupported HTTP method.")
    return tool
