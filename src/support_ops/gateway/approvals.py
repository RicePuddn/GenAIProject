"""Ticket state machine and approval tokens for the multi-agent arm.

Composes the shared `TicketState` (audit log and handoffs) without changing it. A token is
issued only for a route at least as strict as the rule engine's minimum, and is bound to one
ticket, operation, order and amount. Every transition and refusal is audited.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from support_ops.agents.models import ApprovalResult, HandoffPacket, Proposal
from support_ops.gateway.ticket_state import TicketState
from support_ops.helpers.clock import Clock
from support_ops.policy.routing import at_least

TRANSITIONS: dict[str, set[str]] = {
    "open": {"triaged", "awaiting_customer", "awaiting_human"},
    "triaged": {"proposed", "awaiting_customer", "awaiting_human"},
    # proposed → triaged: sent back for a new proposal, or on to the next part of a multi-part ticket.
    # → awaiting_customer from a later state: a part finished, but another part asked the customer something.
    "proposed": {"approved", "triaged", "awaiting_customer", "awaiting_human"},
    "approved": {"executed", "resolved", "triaged", "awaiting_customer", "awaiting_human"},  # resolved: info only
    "executed": {"resolved", "triaged", "awaiting_customer", "awaiting_human"},
    "awaiting_customer": {"awaiting_human"},                 # the question itself failed review
    "awaiting_human": set(),
    "resolved": set(),
}


def _same_amount(a: Any, b: Any) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and Decimal(str(a)) == Decimal(str(b)))


class TicketFlow:
    def __init__(self, ticket_state: TicketState, clock: Clock, token_ttl_seconds: int = 300):
        self.ticket_state, self.clock, self.token_ttl_seconds = ticket_state, clock, token_ttl_seconds
        self.statuses: dict[str, str] = {}
        self.tokens: dict[str, dict[str, Any]] = {}

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        return self.ticket_state.audit_log

    def status(self, ticket_id: str) -> str:
        return self.statuses.get(ticket_id, "open")

    def audit(self, ticket_id: str, kind: str, **details: Any) -> None:
        # `action` is TicketState's dispatch key, so details can never override it.
        self.ticket_state.run(ticket_id, {"type": kind, **details, "action": "audit"})

    def refuse(self, ticket_id: str, kind: str, reason: str, **details: Any) -> None:
        self.audit(ticket_id, kind, outcome="refused", reason=reason, **details)
        raise ValueError(reason)

    def transition(self, ticket_id: str, to: str, **details: Any) -> str:
        current = self.status(ticket_id)
        if to not in TRANSITIONS.get(current, set()):
            self.refuse(ticket_id, "transition", f"transition {current} -> {to} is not allowed",
                        **{"from": current, "to": to})
        self.statuses[ticket_id] = to
        self.audit(ticket_id, "transition", outcome="allowed", **{"from": current, "to": to}, **details)
        return to

    def approve(self, ticket_id: str, proposal: Proposal, minimum: dict[str, Any],
                approval: ApprovalResult | None = None) -> dict[str, Any]:
        """Move a proposed ticket to approved; issue a token when there is something to execute."""
        if self.status(ticket_id) != "proposed":
            self.refuse(ticket_id, "approval", f"cannot approve a ticket in state {self.status(ticket_id)}")
        if not minimum.get("allowed", False) or not at_least(proposal.route, minimum["route"]):
            self.refuse(ticket_id, "approval", f"route {proposal.route} is looser than the minimum "
                        f"{minimum['route']} or the action is not allowed", min_route=minimum)
        if proposal.route == "human":
            self.refuse(ticket_id, "approval", "a human route is handed off, not approved by the system")
        if proposal.route == "approver" and not (
                approval and approval.stage == "pre_execution" and approval.decision == "approve"):
            self.refuse(ticket_id, "approval", "the approver route needs the approver's pre-execution approval")
        self.transition(ticket_id, "approved", route=proposal.route, operation=proposal.action)
        if proposal.action == "none":
            return {"status": "approved", "approval_token": None}
        issued = self.clock.now()
        token = secrets.token_urlsafe(18)
        self.tokens[token] = {
            "ticket_id": ticket_id, "operation": proposal.action, "order_id": proposal.order_id,
            "amount": proposal.amount, "route": proposal.route, "issued_at": issued,
            "payment_id": minimum.get("policy_check", {}).get("payment_id"),
            "expires_at": issued + timedelta(seconds=self.token_ttl_seconds), "used": False,
        }
        # The token itself is never written to the audit log, only what it is bound to.
        self.audit(ticket_id, "token_issued", operation=proposal.action, order_id=proposal.order_id,
                   amount=proposal.amount, route=proposal.route,
                   expires_at=self.tokens[token]["expires_at"].isoformat())
        return {"status": "approved", "approval_token": token,
                "expires_at": self.tokens[token]["expires_at"].isoformat()}

    def redeem(self, token: str, ticket_id: str, operation: str, order_id: str | None, amount: Any) -> dict[str, Any]:
        """Check a token against the write it is presented with and mark it used."""
        grant = self.tokens.get(token)
        details = {"operation": operation, "order_id": order_id, "amount": amount}
        if grant is None:
            self.refuse(ticket_id, "token", "unknown approval token", **details)
        if grant["used"]:
            self.refuse(ticket_id, "token", "approval token already used", **details)
        if (grant["ticket_id"], grant["operation"], grant["order_id"]) != (ticket_id, operation, order_id) \
                or not _same_amount(grant["amount"], amount):
            self.refuse(ticket_id, "token", "approval token is bound to a different ticket, operation, "
                        "order or amount", **details)
        now: datetime = self.clock.now()
        if now > grant["expires_at"]:
            self.refuse(ticket_id, "token", "approval token expired", **details)
        grant["used"] = True
        return grant

    def handoff(self, packet: HandoffPacket) -> dict[str, Any]:
        self.transition(packet.ticket_id, "awaiting_human", reason=packet.reason)
        return self.ticket_state.run(packet.ticket_id, {"action": "handoff", **packet.model_dump(mode="json")})
