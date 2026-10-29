"""A specialist's view of the shared tools: a named subset of `SupportTools` plus tools
only the multi-agent arm has (`route`, and a search-only `policy`). `SupportTools` itself is unchanged."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel

from support_ops.helpers.arguments import NonEmpty, ToolArguments
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


class PolicySearchArguments(ToolArguments):
    """Retrieve the policy clauses most relevant to a query, with their exceptions. scope own (the
    default) keeps clauses from your policy sections; use all when the case crosses sections."""
    action: Literal["search"] = "search"
    query: NonEmpty
    scope: Literal["own", "all"] = "own"


def own_sections(result: dict[str, Any], sections: Sequence[str]) -> dict[str, Any]:
    """Keep the clauses from `sections`, plus the exceptions of the clauses kept, whatever their section."""
    wanted = {section.casefold() for section in sections}
    clauses = result.get("clauses", [])
    kept_ids = {clause["id"] for clause in clauses
                if clause["via"] == "search" and clause["section"].casefold() in wanted}
    kept = [clause for clause in clauses if clause["id"] in kept_ids
            or (clause["via"].startswith("exception_of:") and clause["via"].split(":", 1)[1] in kept_ids)]
    filtered = {**result, "clauses": kept, "scope": "own", "sections": list(sections)}
    if dropped := len(clauses) - len(kept):
        filtered["filtered_out"] = dropped
    if not kept:
        filtered["note"] = ("No clause from your sections matched. If the case crosses into another area, "
                            "search again with scope all.")
    return filtered


def policy_search_tool(tools: SupportTools, sections: Sequence[str] | None = None) -> ExtraTool:
    """The `policy` tool without its `check` action (in the multi-agent arm the `route` tool applies the
    hard rules), limited by default to the specialist's `sections`. Same policy provider as the shared tool.
    The retriever's top-k is filtered afterwards; a native section filter in the retriever would return
    k clauses from the section instead (see docs/multi-agent.md, vector DB notes)."""
    def run(ticket_id: str, args: dict[str, Any]) -> dict[str, Any]:
        result = tools.policy.run({"action": "search", "query": args["query"]})
        if args.get("scope", "own") == "all" or not sections:
            return {**result, "scope": "all"}
        return own_sections(result, sections)

    return ExtraTool(PolicySearchArguments, PolicySearchArguments.__doc__, run)


class Toolbox:
    def __init__(self, tools: SupportTools, names: Sequence[str], extra: dict[str, ExtraTool] | None = None):
        extra = dict(extra or {})
        if unknown := sorted(set(names) - set(tools.argument_models)):
            raise ValueError(f"unknown shared tools: {unknown}")
        if clash := sorted(set(names) & set(extra)):
            raise ValueError(f"extra tools shadow shared tools: {clash}")
        self.tools, self.extra, self.policy = tools, extra, tools.policy
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
