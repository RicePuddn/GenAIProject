"""What the supervisor's reply may say, in customer terms.

Routes, approvals, rule names, thresholds and clause or item IDs stay internal, so the reply
cannot leak or misread them. The reply writer and the approver's reply check both get this outcome.
"""
from __future__ import annotations

from typing import Any

from support_ops.agents.models import Proposal

OFF_TOPIC = ("The message is not about the store. Do not answer it. Briefly say you can help with orders, "
             "shipments, items, payments, returns, refunds and the store's support policies, and invite a "
             "question about those.")
# Ticket states that end the turn resolved, as the customer sees them.
RESOLVED_STATES = {"approved": "resolved", "executed": "resolved"}


def verified_facts(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Successful read and policy results, so the reply rests on what the tools returned."""
    return [{"tool": item["tool"], "result": item["result"]} for item in evidence
            if item["tool"] != "route" and "error" not in item["result"]]


def customer_outcome(*, status: str, evidence: list[dict[str, Any]], proposal: Proposal | None,
                     execution: dict[str, Any] | None, min_route: dict[str, Any] | None,
                     customer_question: str | None, currency: str, off_topic: bool = False,
                     unfound_record: str | None = None) -> dict[str, Any]:
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
    if existing := (min_route or {}).get("existing_requests"):
        # Earlier returns and refunds on this order, in customer terms (no internal IDs).
        outcome["existing_requests"] = [{key: value for key, value in request.items() if key != "id" and value}
                                        for request in existing]
    # Found by code from the records, so the apology does not depend on the model noticing them.
    problems = []
    if (min_route or {}).get("overdue_shipment"):
        problems.append("The shipment is past its estimated delivery date.")
    for request in (min_route or {}).get("existing_requests") or []:
        if request.get("kind") == "return" and request.get("awaiting_the_store"):
            problems.append(f"The return the customer requested on {request['requested']} has not been "
                            "processed by the store yet.")
    if unfound_record:
        # The customer already confirmed it, so asking them to check again would go round in circles.
        problems.append(f"The customer confirmed {unfound_record}, but it is not in our records. Do not ask "
                        "them to check it again.")
    if problems:
        outcome["problems"] = problems
    outcome["tell_customer"] = (OFF_TOPIC if off_topic else
                                _tell_customer(outcome["status"], bool(facts), executed, proposal, customer_question,
                                               bool(problems)))
    return outcome


def _tell_customer(status: str, has_facts: bool, executed: bool, proposal: Proposal | None,
                   customer_question: str | None, has_problems: bool = False) -> str:
    if status == "awaiting_human" and has_facts:
        return ("Share what the facts show that answers the customer (e.g. the current status and "
                "estimated dates). Acknowledge every item in problems, if any, and apologise; likewise "
                "anything else that has gone wrong, such as a damaged item. Then say a support specialist "
                "will review the request and follow up. Promise no outcome, refund, compensation or timeline.")
    if status == "awaiting_human":
        hold = "A support specialist will review the request. Promise nothing else."
        return f"Acknowledge every item in problems and apologise. {hold}" if has_problems else hold
    if status == "awaiting_customer":
        return f"Ask: {customer_question}"
    if executed:
        return f"The {proposal.action} is recorded and complete; nothing is pending review. Say what was recorded."
    return ("Answer the question from the facts. If problems are listed, acknowledge each and apologise; "
            "if none are listed, do not apologise. Nothing was changed on the account; do not mention that.")
