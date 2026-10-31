"""Allowlisted PostgreSQL access; model arguments never contain SQL.

`PostgresReader` is read-only. `PostgresWriter` runs one fixed transaction per approved operation.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

RECORD_KINDS = (
    "customers", "products", "orders", "order_items", "payments",
    "returns", "refunds", "support_tickets",
)

RECORD_PREFIXES = {"refund": "RFD-W", "return": "RET-W", "exchange": "EXC-W", "order_change": "CHG-W"}


class WriteRefused(ValueError):
    """The records at commit time do not allow this write."""


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

    def shipment_events_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return self._select(
            """SELECT e.id, e.shipment_id, s.tracking_number, e.occurred_at,
                      e.status, e.location, e.details
               FROM shipment_events e JOIN shipments s ON s.id = e.shipment_id
               WHERE lower(s.order_id) = lower(%s) ORDER BY e.occurred_at, e.id""",
            order_id,
        )


class PostgresWriter:
    """Applies one approved operation in one transaction, with its idempotency record.

    The order row stays locked until commit, so limits are checked against the records the write will
    change. No customer ownership check yet: when login arrives, it belongs in `apply`, after the lock.
    """

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.environ.get("SUPPORT_OPS_WRITE_DATABASE_URL") \
            or os.environ.get("SUPPORT_OPS_DATABASE_URL")

    def _connect(self) -> psycopg.Connection:
        if not self.dsn:
            raise RuntimeError("set SUPPORT_OPS_WRITE_DATABASE_URL or SUPPORT_OPS_DATABASE_URL to enable writes")
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5,
                               options="-c statement_timeout=10000")

    def replay(self, idempotency_key: str) -> dict[str, Any] | None:
        """The stored request and result of an earlier write with this key."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT request, result FROM backend_writes WHERE idempotency_key = %s", (idempotency_key,))
            return cur.fetchone()

    def apply(self, request: dict[str, Any], *, idempotency_key: str, reason: str, at: datetime,
              payment_id: str | None = None, return_id: str | None = None) -> dict[str, Any]:
        """Run `request` (ticket_id, operation, order_id, amount, items). Raises WriteRefused."""
        operation = request["operation"]
        if operation not in RECORD_PREFIXES:
            raise WriteRefused(f"unsupported write operation: {operation}")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM orders WHERE lower(id) = lower(%s) FOR UPDATE", (request["order_id"],))
            order = cur.fetchone()
            if order is None:
                raise WriteRefused(f"order {request['order_id']} not found")
            # A concurrent retry waited on the lock above, so it is caught here.
            cur.execute("SELECT result FROM backend_writes WHERE idempotency_key = %s", (idempotency_key,))
            if (previous := cur.fetchone()) is not None:
                return {**previous["result"], "replayed": True}
            digest = hashlib.sha256(idempotency_key.encode()).hexdigest()[:12].upper()
            record_id = f"{RECORD_PREFIXES[operation]}-{digest}"
            if operation == "refund":
                payment_id = self._refund(cur, record_id, order, request["amount"], reason, at, payment_id, return_id)
            elif operation == "return":
                self._return(cur, record_id, order, request["items"], reason, at)
            elif operation == "exchange":
                self._order_lines(cur, order, request["items"])
            else:
                self._cancel(cur, order)
            result = {"status": "executed", **request, "order_id": order["id"], "currency": order["currency"],
                      "payment_id": payment_id, "write_id": record_id, "executed_at": at.isoformat()}
            cur.execute(
                "INSERT INTO backend_writes (idempotency_key, ticket_id, operation, order_id, request, result, "
                "created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (idempotency_key, request["ticket_id"], operation, order["id"], Jsonb(request), Jsonb(result), at))
            return result

    @staticmethod
    def _refund(cur: Any, record_id: str, order: dict[str, Any], amount: Any, reason: str, at: datetime,
                payment_id: str | None, return_id: str | None) -> str:
        """A pending refund on the order's captured payment: no money moves here."""
        cur.execute("SELECT id, amount FROM payments WHERE order_id = %s AND status = 'captured'", (order["id"],))
        captured = cur.fetchall()
        ids = [row["id"] for row in captured]
        if payment_id is None and len(ids) == 1:
            payment_id = ids[0]
        if payment_id not in ids:
            raise WriteRefused("refund needs one captured payment on the order")
        cur.execute("SELECT COALESCE(SUM(amount), 0) AS total FROM refunds "
                    "WHERE order_id = %s AND status NOT IN ('failed', 'cancelled', 'rejected')", (order["id"],))
        balance = sum((row["amount"] for row in captured), Decimal(0)) - cur.fetchone()["total"]
        value = Decimal(str(amount))
        if value > balance:
            raise WriteRefused(f"refund {value} exceeds the {balance} balance left on the order")
        if return_id is not None:
            cur.execute("SELECT 1 FROM returns WHERE id = %s AND order_id = %s", (return_id, order["id"]))
            if cur.fetchone() is None:
                raise WriteRefused(f"return {return_id} is not on order {order['id']}")
        cur.execute(
            "INSERT INTO refunds (id, order_id, payment_id, status, amount, reason, requested_at, return_id) "
            "VALUES (%s, %s, %s, 'pending', %s, %s, %s, %s)",
            (record_id, order["id"], payment_id, value, reason, at, return_id))
        return payment_id

    @staticmethod
    def _order_lines(cur: Any, order: dict[str, Any], items: list[dict[str, Any]]) -> None:
        """Every listed line is on this order, in at most the quantity ordered."""
        cur.execute("SELECT id, quantity FROM order_items WHERE order_id = %s", (order["id"],))
        quantities = {row["id"]: row["quantity"] for row in cur.fetchall()}
        if not items:
            raise WriteRefused("this write needs the order items it applies to")
        for line in items:
            if line["quantity"] > quantities.get(line["item_id"], 0):
                raise WriteRefused(f"item {line['item_id']} × {line['quantity']} is not on order {order['id']}")

    def _return(self, cur: Any, record_id: str, order: dict[str, Any], items: list[dict[str, Any]],
                reason: str, at: datetime) -> None:
        self._order_lines(cur, order, items)
        cur.execute(
            "INSERT INTO returns (id, order_id, status, reason, requested_at, evidence_status, inspection_status) "
            "VALUES (%s, %s, 'requested', %s, %s, 'pending', 'pending')", (record_id, order["id"], reason, at))
        cur.executemany("INSERT INTO return_items (return_id, order_item_id, quantity) VALUES (%s, %s, %s)",
                        [(record_id, line["item_id"], line["quantity"]) for line in items])

    @staticmethod
    def _cancel(cur: Any, order: dict[str, Any]) -> None:
        """Cancellation only; any refund it leads to is its own request."""
        if order["status"] != "processing":
            raise WriteRefused(f"order {order['id']} is {order['status']}, so it cannot be cancelled")
        cur.execute("UPDATE orders SET status = 'cancelled' WHERE id = %s", (order["id"],))
        cur.execute("UPDATE order_items SET item_status = 'cancelled' WHERE order_id = %s", (order["id"],))
        cur.execute("UPDATE shipments SET status = 'cancelled' WHERE order_id = %s", (order["id"],))
