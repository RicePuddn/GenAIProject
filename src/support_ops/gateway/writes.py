"""Backend write endpoint for the multi-agent arm.

A write executes only with a valid approval token for exactly that ticket, operation, order,
amount and items. Repeating a request with the same idempotency key returns the first result
instead of writing twice. Every attempt is audited, allowed or refused.
"""
from __future__ import annotations

from typing import Any, Sequence

import psycopg

from support_ops.backend.postgres import PostgresWriter
from support_ops.gateway.approvals import TicketFlow


class BackendWrites:
    def __init__(self, flow: TicketFlow, writer: PostgresWriter):
        self.flow, self.writer = flow, writer

    def execute(self, ticket_id: str, *, token: str, operation: str, order_id: str | None,
                amount: Any = None, items: Sequence[dict[str, Any]] = (), reason: str = "",
                idempotency_key: str) -> dict[str, Any]:
        lines = sorted(items, key=lambda line: line["item_id"])
        write = {"operation": operation, "order_id": order_id, "amount": amount}
        request = {"ticket_id": ticket_id, **write, "items": lines}
        audit = {"idempotency_key": idempotency_key, **write}
        def refused(why: str) -> dict[str, Any]:
            self.flow.audit(ticket_id, "write", outcome="refused", reason=why, **audit)
            return {"status": "refused", "reason": why, **request}

        try:
            if (previous := self.writer.replay(idempotency_key)) is not None:
                if previous["request"] != request:
                    return refused("idempotency key reused for a different write")
                self.flow.audit(ticket_id, "write", outcome="replayed", idempotency_key=idempotency_key,
                                write_id=previous["result"]["write_id"])
                return {**previous["result"], "replayed": True}
            if (status := self.flow.status(ticket_id)) != "approved":
                return refused(f"cannot execute a write for a ticket in state {status}")
            grant = self.flow.redeem(token, ticket_id, operation, order_id, amount, lines)
            result = self.writer.apply(request, idempotency_key=idempotency_key, reason=reason,
                                       at=self.flow.clock.now(), payment_id=grant.get("payment_id"),
                                       return_id=grant.get("return_id"))
        except (ValueError, RuntimeError, psycopg.Error) as exc:
            return refused(str(exc))
        self.flow.transition(ticket_id, "executed", write_id=result["write_id"])
        self.flow.audit(ticket_id, "write", outcome="executed", **audit, write_id=result["write_id"])
        return result
