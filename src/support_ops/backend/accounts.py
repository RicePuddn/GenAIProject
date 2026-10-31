"""The signed-in customer's view of the database, the same for every architecture.

A workflow given a `customer_id` wraps its reader and writer in these, so every path to the data
(tools, policy checks, the multi-agent route tool and gateway, ticket saves) is scoped without any
agent having to remember to scope. Another customer's records read exactly as missing ones do, so
an agent cannot learn that they exist.
"""
from __future__ import annotations

from typing import Any

from support_ops.backend.postgres import PostgresReader, PostgresWriter

# Record kinds linked to their customer through an order.
ORDER_RECORDS = {"order_items", "payments", "returns", "refunds"}


class AccountReader:
    def __init__(self, reader: PostgresReader, customer_id: str):
        self.reader, self.customer_id = reader, customer_id
        self._owned: dict[str, bool] = {}   # order ID (casefolded) → is this customer's

    def owns_order(self, order_id: str) -> bool:
        key = order_id.casefold()
        if key not in self._owned:
            order = self.reader.read("orders", order_id)
            self._owned[key] = not order.get("not_found") and order.get("customer_id") == self.customer_id
        return self._owned[key]

    def read(self, kind: str, record_id: str) -> dict[str, Any]:
        row = self.reader.read(kind, record_id)
        if row.get("not_found") or kind == "products":
            return row
        if kind == "customers":
            visible = row.get("id") == self.customer_id
        elif kind in {"orders", "support_tickets"}:
            visible = row.get("customer_id") == self.customer_id
        elif kind in ORDER_RECORDS:
            visible = bool(row.get("order_id")) and self.owns_order(row["order_id"])
        else:
            visible = False
        return row if visible else {"not_found": True, "kind": kind, "id": record_id}

    def own_orders(self) -> list[dict[str, Any]]:
        """The signed-in customer's own orders."""
        return self.reader.orders_for_customer(self.customer_id)

    # Lookups by order: another customer's order has nothing, as a missing order has nothing.
    def shipments_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return self.reader.shipments_for_order(order_id) if self.owns_order(order_id) else []

    def shipment_events_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return self.reader.shipment_events_for_order(order_id) if self.owns_order(order_id) else []

    def items_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return self.reader.items_for_order(order_id) if self.owns_order(order_id) else []

    def payments_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return self.reader.payments_for_order(order_id) if self.owns_order(order_id) else []

    def returns_for_order(self, order_id: str) -> dict[str, list[dict[str, Any]]]:
        return self.reader.returns_for_order(order_id) if self.owns_order(order_id) else {"returns": [], "refunds": []}


class AccountWriter:
    """Writes as the signed-in customer: their orders only, and their tickets."""

    def __init__(self, writer: PostgresWriter, customer_id: str):
        self.writer, self.customer_id = writer, customer_id

    def replay(self, idempotency_key: str) -> dict[str, Any] | None:
        return self.writer.replay(idempotency_key)

    def apply(self, request: dict[str, Any], **details: Any) -> dict[str, Any]:
        return self.writer.apply(request, **details, customer_id=self.customer_id)

    def save_ticket(self, **ticket: Any) -> dict[str, Any]:
        return self.writer.save_ticket(**ticket, customer_id=self.customer_id)


def for_account(reader: PostgresReader, writer: PostgresWriter,
                customer_id: str | None) -> tuple[PostgresReader, PostgresWriter]:
    """The reader and writer a workflow uses: scoped to `customer_id`, or as given when no one is signed in."""
    if not customer_id:
        return reader, writer
    return AccountReader(reader, customer_id), AccountWriter(writer, customer_id)
