"""In-memory `PostgresWriter` for tests that script the agents; real writes are in test_postgres_writer.py."""
from datetime import datetime
from typing import Any


class FakeWriter:
    def __init__(self):
        self.applied: dict[str, dict[str, Any]] = {}

    def replay(self, idempotency_key: str) -> dict[str, Any] | None:
        return self.applied.get(idempotency_key)

    def apply(self, request: dict[str, Any], *, idempotency_key: str, reason: str, at: datetime,
              payment_id: str | None = None, return_id: str | None = None) -> dict[str, Any]:
        result = {"status": "executed", **request, "currency": "SGD", "payment_id": payment_id,
                  "write_id": f"W-{len(self.applied) + 1:05d}", "executed_at": at.isoformat()}
        self.applied[idempotency_key] = {"request": request, "result": result}
        return result
