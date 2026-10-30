"""Mock backend write endpoint for the multi-agent arm.

A write executes only with a valid approval token for exactly that ticket, operation, order
and amount. Repeating a request with the same idempotency key returns the first result instead
of writing twice. Every attempt is audited, allowed or refused. Nothing touches PostgreSQL.
"""
from __future__ import annotations

from typing import Any

from support_ops.gateway.approvals import TicketFlow


class MockWrites:
    def __init__(self, flow: TicketFlow, currency: str = "SGD"):
        self.flow, self.currency = flow, currency
        self.executed: dict[str, dict[str, Any]] = {}

    def execute(self, ticket_id: str, *, token: str, operation: str, order_id: str | None,
                amount: Any = None, idempotency_key: str) -> dict[str, Any]:
        write = {"operation": operation, "order_id": order_id, "amount": amount}
        request = {"ticket_id": ticket_id, **write}
        if (previous := self.executed.get(idempotency_key)) is not None:
            if {key: previous[key] for key in request} != request:
                self.flow.refuse(ticket_id, "write", "idempotency key reused for a different write",
                                 idempotency_key=idempotency_key)
            self.flow.audit(ticket_id, "write", outcome="replayed", idempotency_key=idempotency_key,
                            write_id=previous["write_id"])
            return {**previous, "replayed": True}
        try:
            if (status := self.flow.status(ticket_id)) != "approved":
                raise ValueError(f"cannot execute a write for a ticket in state {status}")
            self.flow.redeem(token, ticket_id, operation, order_id, amount)
        except ValueError as exc:
            self.flow.audit(ticket_id, "write", outcome="refused", reason=str(exc),
                            idempotency_key=idempotency_key, **write)
            return {"status": "refused", "reason": str(exc), **request}
        result = {
            "status": "executed", "mock": True, **request, "currency": self.currency,
            "write_id": f"MW-{len(self.executed) + 1:05d}", "executed_at": self.flow.clock.now().isoformat(),
        }
        self.executed[idempotency_key] = result
        self.flow.transition(ticket_id, "executed", write_id=result["write_id"])
        self.flow.audit(ticket_id, "write", outcome="executed", idempotency_key=idempotency_key, **write,
                        write_id=result["write_id"])
        return result
