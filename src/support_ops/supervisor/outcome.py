"""What the supervisor's reply may say, in customer terms.

Routes, approvals, rule names, thresholds and clause or item IDs stay internal, so the reply
cannot leak or misread them. The reply writer and the approver's reply check both get this outcome.
"""
from __future__ import annotations

from typing import Any

from support_ops.agents.models import Proposal

# Ticket states that end the turn resolved, as the customer sees them.
RESOLVED_STATES = {"approved": "resolved", "executed": "resolved"}


def verified_facts(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Successful read and policy results, so the reply rests on what the tools returned."""
    return [{"tool": item["tool"], "result": item["result"]} for item in evidence
            if item["tool"] != "route" and "error" not in item["result"]]


def customer_outcome(*, status: str, evidence: list[dict[str, Any]], proposal: Proposal | None,
                     execution: dict[str, Any] | None, min_route: dict[str, Any] | None,
                     customer_question: str | None, currency: str) -> dict[str, Any]:
    facts = verified_facts(evidence)
    names = {item["id"]: item.get("name") for fact in facts if fact["tool"] == "get_items_for_order"
             for item in fact["result"].get("items", [])}
    executed = bool(execution and execution.get("status") == "executed")
    outcome: dict[str, Any] = {"status": RESOLVED_STATES.get(status, status), "facts": facts}
    if proposal and proposal.action != "none":
        request: dict[str, Any] = {"action": proposal.action, "order_id": proposal.order_id}
        if proposal.amount is not None:
            request["amount"] = f"{proposal.amount:.2f} {currency}"
        if proposal.items:
            request["items"] = [{"product": names.get(line.item_id) or "the item", "quantity": line.quantity}
                                for line in proposal.items]
        outcome["request"] = request
    if proposal:
        # What the specialist found, e.g. that a shipment is past its estimated date.
        outcome["specialist_finding"] = proposal.reason
    if (min_route or {}).get("overdue_shipment"):
        # Found by code from the shipment dates, so the apology does not depend on the model's arithmetic.
        outcome["problems"] = ["The shipment is past its estimated delivery date."]
    outcome["tell_customer"] = _tell_customer(outcome["status"], bool(facts), executed, proposal, customer_question)
    return outcome


def _tell_customer(status: str, has_facts: bool, executed: bool, proposal: Proposal | None,
                   customer_question: str | None) -> str:
    if status == "awaiting_human" and has_facts:
        return ("Share what the facts show that answers the customer (e.g. the current status and "
                "estimated dates). Acknowledge every item in problems, if any, and apologise; likewise "
                "anything else that has gone wrong, such as a damaged item. Then say a support specialist "
                "will review the request and follow up. Promise no outcome, refund, compensation or timeline.")
    if status == "awaiting_human":
        return "A support specialist will review the request. Promise nothing else."
    if status == "awaiting_customer":
        return f"Ask: {customer_question}"
    if executed:
        return f"The {proposal.action} is recorded and complete; nothing is pending review. Say what was recorded."
    return ("Answer the question from the facts. If problems are listed, acknowledge each and apologise; "
            "if none are listed, do not apologise. Nothing was changed on the account; do not mention that.")
