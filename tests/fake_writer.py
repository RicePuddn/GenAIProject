"""In-memory `PostgresWriter` for tests that script the agents; real writes are in test_postgres_writer.py."""
from datetime import datetime
from typing import Any


class FakeWriter:
    def __init__(self):
        self.applied: dict[str, dict[str, Any]] = {}
        self.ticket_saves: list[dict[str, Any]] = []

    def save_ticket(self, **ticket: Any) -> dict[str, Any]:
        self.ticket_saves.append(ticket)
        return {"id": ticket["ticket_id"], "status": ticket["status"],
                "customer_id": ticket.get("customer_id") or "dummy_00", "order_id": ticket.get("order_id")}

    def replay(self, idempotency_key: str) -> dict[str, Any] | None:
        return self.applied.get(idempotency_key)

    def apply(self, request: dict[str, Any], *, idempotency_key: str, reason: str, at: datetime,
              payment_id: str | None = None, return_id: str | None = None,
              customer_id: str | None = None) -> dict[str, Any]:
        result = {"status": "executed", **request, "currency": "SGD", "payment_id": payment_id,
                  "write_id": f"W-{len(self.applied) + 1:05d}", "executed_at": at.isoformat()}
        self.applied[idempotency_key] = {"request": request, "result": result}
        return result
