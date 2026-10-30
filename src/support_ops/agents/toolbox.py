"""A specialist's view of the shared tools: a named subset of `SupportTools` plus tools
only the multi-agent arm has (`route`, and the policy provider's `policy` tool scoped to the specialist).
`SupportTools` itself is unchanged."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from support_ops.policy.provider import ScopedPolicyTool
from support_ops.systems.tools import SupportTools, validate_arguments

ORDERS_SHIPPING_TOOLS = (
    "read_record", "get_shipments_for_order", "get_shipment_events_for_order",
    "get_items_for_order", "policy",
)
# Shipments are included for the delivery date the return window is counted from.
RETURNS_REFUNDS_TOOLS = (
    "read_record", "get_items_for_order", "get_payments_for_order", "get_returns_for_order",
    "get_shipments_for_order", "policy",
)


@dataclass(frozen=True)
class ExtraTool:
    arguments: type[BaseModel]
    description: str
    run: Callable[[str, dict[str, Any]], dict[str, Any]]   # (ticket_id, validated arguments)


def returns_lookup_tool(tools: SupportTools, checked: set[str]) -> ExtraTool:
    """`get_returns_for_order`, recording which orders' returns the specialist has seen, so the
    `route` tool only shows them again for orders it has not looked at."""
    schema = next(item["function"] for item in tools.schemas if item["function"]["name"] == "get_returns_for_order")

    def run(ticket_id: str, args: dict[str, Any]) -> dict[str, Any]:
        result = tools.execute(ticket_id, "get_returns_for_order", args)
        checked.add(args["order_id"])
        return result

    return ExtraTool(tools.argument_models["get_returns_for_order"], schema["description"], run)


class Toolbox:
    def __init__(self, tools: SupportTools, names: Sequence[str],
                 extra: dict[str, ExtraTool | ScopedPolicyTool] | None = None):
        extra = dict(extra or {})
        if unknown := sorted(set(names) - set(tools.argument_models)):
            raise ValueError(f"unknown shared tools: {unknown}")
        if clash := sorted(set(names) & set(extra)):
            raise ValueError(f"extra tools shadow shared tools: {clash}")
        self.tools, self.extra = tools, extra
        # The provider's scoped tool carries its own prompt text and planner overview.
        self.policy = extra["policy"] if isinstance(extra.get("policy"), ScopedPolicyTool) else tools.policy
        self.argument_models: dict[str, type[BaseModel]] = {
            **{name: tools.argument_models[name] for name in names},
            **{name: tool.arguments for name, tool in extra.items()},
        }
        shared = {schema["function"]["name"]: schema for schema in tools.schemas}
        self.schemas = [shared[name] for name in names] + [
            {"type": "function", "function": {
                "name": name, "description": tool.description, "parameters": tool.arguments.model_json_schema(),
            }}
            for name, tool in extra.items()
        ]

    @property
    def names(self) -> list[str]:
        return list(self.argument_models)

    def validate(self, name: str, arguments: Any) -> dict[str, Any]:
        return validate_arguments(self.argument_models, name, arguments)

    def execute(self, ticket_id: str, name: str, arguments: Any) -> dict[str, Any]:
        args = self.validate(name, arguments)
        if name in self.extra:
            return self.extra[name].run(ticket_id, args)
        return self.tools.execute(ticket_id, name, args)
