"""The supervisor and the two specialists, shared by the multi-agent setups (S3, S4, S7, S8).

Each specialist gets a subset of the one tool registry; only the specialists call data tools. The
supervisor's one tool, ask_specialist, runs a specialist on one part of the message: the agentic setups
run it as an agent that writes for itself, the graph setups as the fixed stage sequence.
"""
from __future__ import annotations

from typing import Any, Callable, Literal

from SupportOpsBot.runtime import prompts
from SupportOpsBot.runtime.setups.base import STRICTNESS, Turn
from SupportOpsBot.runtime.shared.policy.provider import PolicyProvider
from SupportOpsBot.runtime.shared.strategies import Agent
from SupportOpsBot.runtime.shared.tools import Tool
from SupportOpsBot.services.structure.structure import AskSpecialistArguments

SpecialistName = Literal["orders_shipping", "returns_refunds"]
# Each specialist's own part of the policy, by policy mode: RAG section names, or skill folders. Its policy
# tool keeps to it unless a call says scope all.
POLICY_SCOPE: dict[str, dict[str, tuple[str, ...]]] = {
    "rag": {"orders_shipping": ("Orders and shipping",), "returns_refunds": ("Returns and refunds",)},
    "skills": {"orders_shipping": ("orders-and-shipping",), "returns_refunds": ("returns-and-refunds",)},
}
# Shipments are in returns_refunds for the delivery date a return window is counted from.
SPECIALIST_TOOLS: dict[str, tuple[str, ...]] = {
    "orders_shipping": ("get_order", "get_customer_orders", "get_shipments", "get_tracking_events",
                        "get_order_items", "policy"),
    "returns_refunds": ("get_order", "get_customer_orders", "get_order_items", "get_payments",
                        "get_returns_and_refunds", "get_shipments", "policy"),
}


def supervisor(turn: Turn, control: str) -> Agent:
    return Agent("supervisor", "supervisor",
                 prompts.load("supervisor", control=prompts.load("control", f"supervisor_{control}")))


def specialist(turn: Turn, name: str, control: str) -> tuple[Agent, PolicyProvider]:
    """The specialist, and its scoped policy tool, which its toolset must use."""
    policy = turn.policy.scoped(POLICY_SCOPE[turn.ctx.cfg["policy"]["mode"]][name])
    prompt = prompts.load("specialists", "common", title=name.replace("_", " and "), today=turn.today,
                          scope=prompts.load("specialists", name), facts=turn.facts(policy),
                          control=prompts.load("control", control))
    return Agent(name, "specialist", prompt, policy.planner_context()), policy


def specialist_brief(turn: Turn, request: str) -> dict[str, Any]:
    return {**turn.brief, "your_task": request}


def ask_tool(run: Callable[[str, str], dict[str, Any]], parts: list[dict[str, Any]]) -> Tool:
    """ask_specialist, recording each report in `parts`."""
    def ask(args: dict[str, Any]) -> dict[str, Any]:
        report = run(args["specialist"], args["request"])
        parts.append({"specialist": args["specialist"], "request": args["request"], **report})
        return {"specialist": args["specialist"], **report}

    from SupportOpsBot.services.tools.ask_specialist import create_tool
    return create_tool(ask)


def report(outcome: str, note: str | None, writes: list[dict[str, Any]]) -> dict[str, Any]:
    """What the supervisor is told about a part: how it ended, the specialist's note and the writes made."""
    return {"outcome": outcome, "note": note,
            "writes": [{key: write.get(key) for key in ("operation", "order_id", "amount", "items")}
                       for write in writes]}


def strictest(parts: list[dict[str, Any]]) -> str:
    return max((part["outcome"] for part in parts if part.get("outcome") in STRICTNESS),
               key=STRICTNESS.index, default="resolved")
