"""Allowlisted PostgreSQL reads; model arguments never contain SQL."""
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
        return self._select(
            "SELECT * FROM shipments WHERE lower(order_id) = lower(%s) "
            "ORDER BY shipped_at NULLS LAST, id",
            order_id,
        )

    def items_for_order(self, order_id: str) -> list[dict[str, Any]]:
        """Order lines with the product each one is for."""
        return self._select(
            """SELECT i.id, i.product_id, p.sku, p.name, p.category, i.quantity, i.unit_price,
                      i.item_status
               FROM order_items i JOIN products p ON p.id = i.product_id
               WHERE lower(i.order_id) = lower(%s) ORDER BY i.id""",
            order_id,
        )

    def payments_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return self._select(
            "SELECT * FROM payments WHERE lower(order_id) = lower(%s) ORDER BY id", order_id)

    def returns_for_order(self, order_id: str) -> dict[str, list[dict[str, Any]]]:
        """Returns and refunds together, since a refund usually follows a return."""
        return {
            "returns": self._select(
                "SELECT * FROM returns WHERE lower(order_id) = lower(%s) ORDER BY requested_at, id",
                order_id),
            "refunds": self._select(
                "SELECT * FROM refunds WHERE lower(order_id) = lower(%s) ORDER BY requested_at, id",
                order_id),
        }

    def shipment_events_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return self._select(
            """SELECT e.id, e.shipment_id, s.tracking_number, e.occurred_at,
                      e.status, e.location, e.details
               FROM shipment_events e JOIN shipments s ON s.id = e.shipment_id
               WHERE lower(s.order_id) = lower(%s) ORDER BY e.occurred_at, e.id""",
            order_id,
        )
