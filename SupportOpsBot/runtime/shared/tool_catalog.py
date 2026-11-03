"""Canonical names and classifications for tools visible in traces and agent toolsets."""
from __future__ import annotations

READ_TOOLS = ("get_customer_orders", "get_order", "get_order_items", "get_shipments", "get_tracking_events",
              "get_payments", "get_returns_and_refunds")
POLICY_TOOLS = ("policy",)
WRITE_OPERATION_BY_TOOL = {
    "create_return_request": "return_request",
    "request_refund": "refund",
    "cancel_order": "cancel",
}
WRITE_TOOLS = tuple(WRITE_OPERATION_BY_TOOL)
REVIEW_TOOLS = ("request_review",)
WRITE_ACTIONS = frozenset(("backend_write", *WRITE_OPERATION_BY_TOOL))
WRITE_REVIEW_TOOLS = frozenset((*WRITE_ACTIONS, *REVIEW_TOOLS))
GRAPH_TRACE_TOOLS = frozenset(("backend_write",))
GATHER_TOOLS = (*READ_TOOLS, *POLICY_TOOLS)
ALL_TOOLS = (*GATHER_TOOLS, *WRITE_TOOLS, *REVIEW_TOOLS)
LOOKUP_CATEGORIES = (("get_tracking_events", "delivery"), ("get_shipments", "delivery"),
                     ("get_returns_and_refunds", "returns"), ("get_payments", "payment"))
def operation_for_tool(name: str, arguments: dict | None = None) -> str | None:
    return (arguments or {}).get("operation") or WRITE_OPERATION_BY_TOOL.get(name)
