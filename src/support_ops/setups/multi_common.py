"""The supervisor and the two specialists, shared by the multi-agent setups (S3, S4, S7, S8).

Each specialist gets a subset of the one tool registry; only the specialists call data tools. The
supervisor's one tool, ask_specialist, runs a specialist on one part of the message: the agentic setups
run it as an agent that writes for itself, the graph setups as the fixed stage sequence.
"""
from __future__ import annotations

from typing import Any, Callable, Literal

from support_ops import prompts
from support_ops.setups.base import STRICTNESS, Turn
from support_ops.shared.strategies import Agent
from support_ops.shared.tools import NonEmpty, Tool, ToolArguments

SpecialistName = Literal["orders_shipping", "returns_refunds"]
# Shipments are in returns_refunds for the delivery date a return window is counted from.
SPECIALIST_TOOLS: dict[str, tuple[str, ...]] = {
    "orders_shipping": ("read_record", "get_orders_for_customer", "get_shipments_for_order",
                        "get_shipment_events_for_order", "get_items_for_order", "policy"),
    "returns_refunds": ("read_record", "get_orders_for_customer", "get_items_for_order", "get_payments_for_order",
                        "get_returns_for_order", "get_shipments_for_order", "policy"),
}


class AskSpecialistArguments(ToolArguments):
    """Ask a specialist to handle one part of the customer's message, naming the order IDs. It checks the
    records and policy and reports back how its part ended (outcome and note) and any writes made."""
    specialist: SpecialistName
    request: NonEmpty


def supervisor(turn: Turn, control: str) -> Agent:
    return Agent("supervisor", "supervisor",
                 prompts.load("supervisor", control=prompts.load("control", f"supervisor_{control}")))


def specialist(turn: Turn, name: str, control: str) -> Agent:
    prompt = prompts.load("specialists", "common", title=name.replace("_", " and "), today=turn.today,
                          scope=prompts.load("specialists", name), facts=turn.facts(),
                          control=prompts.load("control", control))
    return Agent(name, "specialist", prompt, turn.policy.planner_context())


def specialist_brief(turn: Turn, request: str) -> dict[str, Any]:
    return {**turn.brief, "your_task": request}


def ask_tool(run: Callable[[str, str], dict[str, Any]], parts: list[dict[str, Any]]) -> Tool:
    """ask_specialist, recording each report in `parts`."""
    def ask(args: dict[str, Any]) -> dict[str, Any]:
        report = run(args["specialist"], args["request"])
        parts.append({"specialist": args["specialist"], "request": args["request"], **report})
        return {"specialist": args["specialist"], **report}

    return Tool("ask_specialist", AskSpecialistArguments, AskSpecialistArguments.__doc__, ask, data=False)


def report(outcome: str, note: str | None, writes: list[dict[str, Any]]) -> dict[str, Any]:
    """What the supervisor is told about a part: how it ended, the specialist's note and the writes made."""
    return {"outcome": outcome, "note": note,
            "writes": [{key: write.get(key) for key in ("operation", "order_id", "amount", "items")}
                       for write in writes]}


def strictest(parts: list[dict[str, Any]]) -> str:
    return max((part["outcome"] for part in parts if part.get("outcome") in STRICTNESS),
               key=STRICTNESS.index, default="resolved")
