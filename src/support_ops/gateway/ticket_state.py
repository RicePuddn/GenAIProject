"""Request-local simulation of approvals, audit events and human handoffs.

Tokens are opaque simulation artifacts. No production payment or database write
is performed, and state is retained only for the lifetime of this instance.
"""
from __future__ import annotations

import secrets
from datetime import timedelta
from typing import Any

from support_ops.helpers.clock import Clock
from support_ops.policy.rules import Rules


class TicketState:
    def __init__(self, clock: Clock, token_ttl_seconds: int = 300, rules: Rules | None = None):
        self.clock = clock
        self.token_ttl_seconds = token_ttl_seconds
        self.rules = rules or Rules.load()
        self.audit_log: list[dict[str, Any]] = []
        self.writes: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.handoffs: list[dict[str, Any]] = []

    def run(self, ticket_id: str, args: dict[str, Any]) -> dict[str, Any]:
        action = args.get("action")
        event = {**args, "ticket_id": ticket_id, "at": self.clock.now().isoformat()}
        if action == "handoff":
            self.handoffs.append(event)
            self.audit_log.append({**event, "event": "handoff"})
            return {"status": "awaiting_human", "handoff_packet": event}
        if action == "audit":
            self.audit_log.append({**event, "event": "audit"})
            return {"status": "audited"}
        if action != "write":
            raise ValueError("unsupported ticket-state action")

        operation, record_id = args.get("operation"), args.get("record_id")
        if not operation or not record_id:
            raise ValueError("write requires operation and record_id")
        if operation != "refund":
            return self.run(ticket_id, {**args, "action": "handoff", "reason": "unsupported mock write operation"})
        decision = self.rules.check(operation, args.get("amount"), args.get("facts"))
        if not decision["allowed"]:
            return self.run(ticket_id, {
                **args, "action": "handoff",
                "reason": decision.get("reason", "amount exceeds mock approval limit"),
            })
        key = (ticket_id, operation, record_id)
        if key in self.writes:
            previous = self.writes[key]
            if previous["amount"] != args["amount"]:
                raise ValueError("conflicting amount for an existing mock write")
            return previous
        issued_at = self.clock.now()
        result = {
            "status": "approved_mock_write", "mock": True,
            "approval_token": secrets.token_urlsafe(18), "operation": operation,
            "record_id": record_id, "amount": args["amount"], "currency": self.rules.currency,
            "issued_at": issued_at.isoformat(),
            "expires_at": (issued_at + timedelta(seconds=self.token_ttl_seconds)).isoformat(),
        }
        self.writes[key] = result
        self.audit_log.append({**event, "event": "write", "approval_token": result["approval_token"]})
        return result
