"""When the supervisor hands a ticket to a human, and what the operator receives.

The triggers are deterministic: they read the triage, never ask the model again. Handoffs from
later steps (route human, approver escalation, a refused write, a budget hit) pass their own
reason to `handoff_packet`.
"""
from __future__ import annotations

import re
from typing import Any

from support_ops.agents.models import HandoffPacket, Proposal, Triage

HOLD_REPLY = "I’ve referred your request to a support specialist for review."
# Handoffs from these triggers get the fixed hold reply with no further LLM calls.
FIXED_REPLY_TRIGGERS = ("guard", "budget")


def triage_escalation(triage: Triage, confidence_min: float) -> str | None:
    """The reason to hand off straight after triage, or None to go on to a specialist."""
    if triage.wants_human:
        return "customer asked for a human"
    if triage.duplicate_of:
        return f"duplicate of open ticket {triage.duplicate_of}"
    if triage.off_topic:
        return None   # not a support request: the reply redirects the customer, no person needed
    if triage.specialist is None:
        return "no specialist covers this request"
    if triage.confidence < confidence_min:
        return f"low triage confidence ({triage.confidence:.2f})"
    return None


def handoff_packet(*, ticket_id: str, reason: str, customer_message: str, triage: Triage | None,
                   proposal: Proposal | None, min_route: dict[str, Any] | None,
                   evidence: list[dict[str, Any]], execution: dict[str, Any] | None) -> HandoffPacket:
    """Summary, actions taken, cited clauses and a recommendation for the operator."""
    actions = [f"{item['agent']}: {item['tool']}" + (" (error)" if "error" in item["result"] else "")
               for item in evidence]
    if execution:
        actions.append(f"write {execution['status']}: {execution.get('operation')}")
    return HandoffPacket(
        ticket_id=ticket_id, reason=reason,
        summary=f"{triage.intent if triage else 'request'}: {customer_message[:500]}",
        actions_taken=actions, cited_clauses=proposal.cited_clauses if proposal else [],
        recommendation=(f"Specialist proposes {proposal.action} via {proposal.route}: {proposal.reason}"
                        if proposal else "Review the request; the automated flow stopped before a proposal."),
        proposal=proposal, min_route=min_route,
    )


# A record the customer named that the database does not have is a question for the customer, not a
# human: they may have mistyped it. Only when they give the same ID again does it go to a person.
RECORD_NAMES = {"orders": "order", "customers": "customer account", "products": "product",
                "order_items": "order item", "payments": "payment", "returns": "return",
                "refunds": "refund", "support_tickets": "ticket"}
ORDER_NOT_FOUND = re.compile(r"\border (\S+) not found\b")


def missing_records(evidence: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(record name, ID) for each record a tool looked for and did not find, in order, once each."""
    found: list[tuple[str, str]] = []
    for item in evidence:
        result = item["result"]
        if item["tool"] == "read_record" and result.get("not_found"):
            found.append((RECORD_NAMES.get(result.get("kind"), "record"), str(result.get("id"))))
        elif item["tool"] == "route" and (match := ORDER_NOT_FOUND.search(str(result.get("error", "")))):
            found.append(("order", match.group(1).rstrip(".:,;")))
    return list(dict.fromkeys(found))


def customer_insists(history: list[dict[str, str]], record_id: str) -> bool:
    """The customer already gave this ID in an earlier turn, so a mistyped number is unlikely."""
    return any(turn["role"] == "customer" and record_id in turn["content"] for turn in history)


def confirm_question(missing: list[tuple[str, str]]) -> str:
    name, record_id = missing[0]
    return f"We couldn't find {name} {record_id}. Could you check the number and send it again?"
