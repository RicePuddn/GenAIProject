"""The ticket's audit log for this turn, and the `support_tickets` row saved in code at the end of every turn.

No agent saves the ticket: the pipeline does, the same way in every setup, from what the turn recorded.
"""
from __future__ import annotations

from typing import Any

import psycopg

from support_ops.shared.clock import Clock
from support_ops.shared.context import TurnContext
from support_ops.shared.db import PostgresWriter

# The ticket category a turn's first write gives it.
WRITE_CATEGORIES = {"refund": "refund", "return_request": "returns", "cancel": "cancellation"}
# Otherwise, the first of these lookups the turn made.
LOOKUP_CATEGORIES = (("get_shipment_events_for_order", "delivery"), ("get_shipments_for_order", "delivery"),
                     ("get_returns_for_order", "returns"), ("get_payments_for_order", "payment"))


class TicketLog:
    """Request-local audit events, retained only for this turn."""

    def __init__(self, ticket_id: str, clock: Clock):
        self.ticket_id, self.clock = ticket_id, clock
        self.events: list[dict[str, Any]] = []

    def audit(self, kind: str, **details: Any) -> None:
        self.events.append({**details, "type": kind, "ticket_id": self.ticket_id, "at": self.clock.now().isoformat()})


def category(evidence: list[dict[str, Any]]) -> str:
    for item in evidence:
        if item["tool"] == "backend_write" and item["result"].get("status") == "executed":
            return WRITE_CATEGORIES.get(item["arguments"].get("operation"), "other")
    tools = {item["tool"] for item in evidence if "error" not in item["result"]}
    return next((name for tool, name in LOOKUP_CATEGORIES if tool in tools), "other")


def confirmed_order(evidence: list[dict[str, Any]]) -> str | None:
    """An order a write or a lookup found, never one the customer only claimed. The writer ignores an
    ID that is not in the database."""
    for item in evidence:
        if item["tool"] == "backend_write" and item["result"].get("status") == "executed":
            return item["result"]["order_id"]
    for item in evidence:
        result = item["result"]
        if item["tool"] == "read_record" and result.get("id") and not result.get("not_found") \
                and str(result["id"]).upper().startswith("ORD"):
            return result["id"]
    return None


def save_ticket(ctx: TurnContext, writer: PostgresWriter, *, first_message: str, status: str, reply: str,
                handoff_packet: dict[str, Any] | None) -> None:
    """Create or update this conversation's ticket. A failed save is logged: it must not cost the customer
    their reply."""
    try:
        row = writer.save_ticket(
            ticket_id=ctx.ticket_id, at=ctx.clock.now(), status=status, category=category(ctx.evidence),
            priority="normal", subject=" ".join(first_message.split())[:200], customer_message=first_message,
            order_id=confirmed_order(ctx.evidence), resolution=reply if status == "resolved" else None,
            handoff_packet=handoff_packet)
        ctx.log("ticket_saved", status=row["status"], order_id=row.get("order_id"))
    except (psycopg.Error, RuntimeError, ValueError, TypeError) as exc:
        ctx.log("ticket_save_failed", error=str(exc)[:300])
