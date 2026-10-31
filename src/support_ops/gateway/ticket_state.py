"""Request-local audit events and human handoffs, retained only for the lifetime of this instance."""
from __future__ import annotations

from typing import Any

from support_ops.helpers.clock import Clock
from support_ops.policy.rules import Rules


class TicketState:
    def __init__(self, clock: Clock, rules: Rules | None = None):
        self.clock = clock
        self.rules = rules or Rules.load()
        self.audit_log: list[dict[str, Any]] = []
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
        raise ValueError("unsupported ticket-state action")
