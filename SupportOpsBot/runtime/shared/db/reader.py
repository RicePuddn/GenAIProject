"""Allowlisted, read-only PostgreSQL access; model arguments never contain SQL."""
from __future__ import annotations

import os
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

RECORD_KINDS = (
    "customers", "products", "orders", "order_items", "payments",
    "returns", "refunds", "support_tickets",
)
# The customer on a ticket when no account is signed in: naming an order does not prove who is chatting.
UNKNOWN_CUSTOMER = "dummy_00"


class PostgresReader:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.environ.get("SUPPORT_OPS_DATABASE_URL")

    def _select(self, query: Any, record_id: str) -> list[dict[str, Any]]:
        if not self.dsn:
            raise RuntimeError("set SUPPORT_OPS_DATABASE_URL to enable PostgreSQL reads")
        with psycopg.connect(
            self.dsn, row_factory=dict_row, connect_timeout=5,
            options="-c default_transaction_read_only=on -c statement_timeout=10000",
        ) as conn, conn.cursor() as cur:
            cur.execute(query, (record_id,))
            return cur.fetchall()

    def read(self, kind: str, record_id: str) -> dict[str, Any]:
        if kind not in RECORD_KINDS:
            raise ValueError("unsupported record type")
        rows = self._select(
            sql.SQL("SELECT * FROM {} WHERE lower(id) = lower(%s)").format(sql.Identifier(kind)),
            record_id,
        )
        return rows[0] if rows else {"not_found": True, "kind": kind, "id": record_id}

    def shipments_for_order(self, order_id: str) -> list[dict[str, Any]]:
        shipments = self._select(
            "SELECT * FROM shipments WHERE lower(order_id) = lower(%s) "
            "ORDER BY shipped_at NULLS LAST, id",
            order_id,
        )
        items = self._select(
            "SELECT si.* FROM shipment_items si JOIN shipments s ON s.id = si.shipment_id "
            "WHERE lower(s.order_id) = lower(%s) ORDER BY si.shipment_id, si.order_item_id", order_id)
        for shipment in shipments:
            shipment["items"] = [item for item in items if item["shipment_id"] == shipment["id"]]
        return shipments

    def items_for_order(self, order_id: str) -> list[dict[str, Any]]:
        """Order lines with the product each one is for."""
        return self._select(
            """SELECT i.id, i.product_id, p.sku, p.name, p.category, i.quantity, i.unit_price,
                      i.item_status, i.final_sale, i.paid_amount, p.product_type
               FROM order_items i JOIN products p ON p.id = i.product_id
               WHERE lower(i.order_id) = lower(%s) ORDER BY i.id""",
            order_id,
        )

    def payments_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return self._select(
            "SELECT * FROM payments WHERE lower(order_id) = lower(%s) ORDER BY id", order_id)

    def returns_for_order(self, order_id: str) -> dict[str, list[dict[str, Any]]]:
        """Returns and refunds together, since a refund usually follows a return."""
        returns = self._select(
            "SELECT * FROM returns WHERE lower(order_id) = lower(%s) ORDER BY requested_at, id", order_id)
        items = self._select(
            "SELECT ri.* FROM return_items ri JOIN returns r ON r.id = ri.return_id "
            "WHERE lower(r.order_id) = lower(%s) ORDER BY ri.return_id, ri.order_item_id", order_id)
        for request in returns:
            request["items"] = [item for item in items if item["return_id"] == request["id"]]
        return {"returns": returns, "refunds": self._select(
            "SELECT * FROM refunds WHERE lower(order_id) = lower(%s) ORDER BY requested_at, id", order_id)}

    def orders_for_customer(self, customer_id: str) -> list[dict[str, Any]]:
        """A customer's latest orders, newest first, each with its product names, so an agent can find the
        order a customer means."""
        return self._select(
            """SELECT o.id, o.placed_at, o.status, o.total, o.currency,
                      string_agg(p.name, ', ' ORDER BY i.id) AS items
               FROM orders o
               LEFT JOIN order_items i ON i.order_id = o.id LEFT JOIN products p ON p.id = i.product_id
               WHERE o.customer_id = %s
               GROUP BY o.id ORDER BY o.placed_at DESC, o.id LIMIT 20""",
            customer_id,
        )

    def customers(self) -> list[dict[str, Any]]:
        """The accounts a demo can sign in as, without the placeholder customer."""
        return self._select("SELECT id, first_name, last_name, email, loyalty_tier FROM customers "
                            "WHERE id <> %s ORDER BY id", UNKNOWN_CUSTOMER)

    def shipment_events_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return self._select(
            """SELECT e.id, e.shipment_id, s.tracking_number, e.occurred_at,
                      e.status, e.location, e.details
               FROM shipment_events e JOIN shipments s ON s.id = e.shipment_id
               WHERE lower(s.order_id) = lower(%s) ORDER BY e.occurred_at, e.id""",
            order_id,
        )
