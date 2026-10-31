"""The `support_tickets` row for a multi-agent ticket: code reads the workflow state, no LLM call."""
from __future__ import annotations

from typing import Any

from support_ops.agents.models import Case

SPECIALIST_TEAMS = {"orders_shipping": "order_support", "returns_refunds": "returns_team"}
FINAL_STATUSES = {"resolved", "awaiting_customer", "awaiting_human"}


def ticket_status(flow_status: str) -> str:
    """Outcomes as they are; every state still in the workflow is in progress."""
    return flow_status if flow_status in FINAL_STATUSES else "in_progress"


def confirmed_order(parts: list[dict[str, Any]], proposal: Any, evidence: list[dict[str, Any]]) -> str | None:
    """An order from a proposal or a lookup, never one the customer only claimed. The writer ignores
    an ID that is not in the database."""
    proposed = [part["proposal"] for part in parts if part.get("proposal")] + ([proposal] if proposal else [])
    for item in proposed:
        if item.order_id:
            return item.order_id
    for item in evidence:
        result = item["result"]
        if item["tool"] == "read_record" and result.get("id") and not result.get("not_found") \
                and str(result["id"]).upper().startswith("ORD"):
            return result["id"]
    return None


def assigned_team(case: Case | None, parts: list[dict[str, Any]]) -> str | None:
    """The team of the first part needing a person, else of the first part; None leaves it to the category."""
    if case and case.category == "payment":
        return "payments_team"
    waiting = [part for part in parts if part["status"] == "awaiting_human"]
    return SPECIALIST_TEAMS[(waiting or parts)[0]["specialist"]] if parts else None


def ticket_fields(*, flow_status: str, case: Case | None, customer_message: str, parts: list[dict[str, Any]],
                  proposal: Any, evidence: list[dict[str, Any]], reply: str | None,
                  handoff_packet: dict[str, Any] | None) -> dict[str, Any]:
    status = ticket_status(flow_status)
    return {"status": status, "category": case.category if case else "other",
            "priority": case.urgency if case else "normal",
            "subject": case.issue[:200] if case else "Held for review before classification",
            "customer_message": customer_message, "assigned_team": assigned_team(case, parts),
            "order_id": confirmed_order(parts, proposal, evidence),
            "resolution": reply if status == "resolved" else None, "handoff_packet": handoff_packet}
