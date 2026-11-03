"""`PostgresWriter` runs one fixed transaction per approved operation, and saves support tickets."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from SupportOpsBot.runtime.shared.db.reader import UNKNOWN_CUSTOMER

# The `support_tickets.category` values, plus other.
TicketCategory = Literal["delivery", "returns", "refund", "payment", "order_change", "cancellation", "product",
                         "account", "other"]
TicketStatus = Literal["in_progress", "resolved", "awaiting_customer", "awaiting_human"]
TicketPriority = Literal["low", "normal", "high"]
CATEGORY_TEAMS = {"payment": "payments_team", "refund": "returns_team", "returns": "returns_team"}
RECORD_PREFIXES = {"refund": "RFD-W", "return": "RET-W", "exchange": "EXC-W", "order_change": "CHG-W"}


def json_dumps(value: Any) -> str:
    """JSON for a JSONB column. A handoff packet can carry raw records, whose dates and amounts are
    datetime and Decimal values; they are stored as text."""
    return json.dumps(value, default=str)


class WriteRefused(ValueError):
    """The records at commit time do not allow this write."""


class PostgresWriter:
    """Applies one approved operation in one transaction, with its idempotency record.

    The order row stays locked until commit, so limits are checked against the records the write will
    change. With a signed-in customer, `apply` checks the order is theirs after taking the lock.
    """

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.environ.get("SUPPORT_OPS_WRITE_DATABASE_URL") \
            or os.environ.get("SUPPORT_OPS_DATABASE_URL")

    def _connect(self) -> psycopg.Connection:
        if not self.dsn:
            raise RuntimeError("set SUPPORT_OPS_WRITE_DATABASE_URL or SUPPORT_OPS_DATABASE_URL to enable writes")
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5,
                               options="-c statement_timeout=10000")

    def save_ticket(self, *, ticket_id: str, status: str, category: str, priority: str, subject: str,
                    customer_message: str, assigned_team: str | None = None, order_id: str | None = None,
                    resolution: str | None = None, handoff_packet: dict[str, Any] | None = None,
                    at: datetime, customer_id: str | None = None) -> dict[str, Any]:
        """Create or update one conversation's ticket; a ticket keeps its first message, creation time and
        customer. The ticket is the signed-in `customer_id`'s, else `UNKNOWN_CUSTOMER` (no account given:
        naming an order does not prove who is chatting). An `order_id` not in the database, or not that
        customer's, is ignored. Another customer's ticket is never updated: that raises WriteRefused."""
        team = assigned_team or CATEGORY_TEAMS.get(category, "order_support")
        customer = customer_id or UNKNOWN_CUSTOMER
        with self._connect() as conn, conn.cursor() as cur:
            order = None
            if order_id:
                cur.execute("SELECT id FROM orders WHERE lower(id) = lower(%s) AND (%s::text IS NULL "
                            "OR customer_id = %s)", (order_id, customer_id, customer_id))
                order = cur.fetchone()
            cur.execute(
                """INSERT INTO support_tickets (id, customer_id, order_id, created_at, channel, category, priority,
                       status, subject, customer_message, resolution, assigned_team, handoff_packet, updated_at)
                   VALUES (%(id)s, %(customer)s, %(order)s, %(at)s, 'chat', %(category)s, %(priority)s, %(status)s,
                       %(subject)s, %(message)s, %(resolution)s, %(team)s, %(packet)s, %(at)s)
                   ON CONFLICT (id) DO UPDATE SET
                       order_id = COALESCE(support_tickets.order_id, EXCLUDED.order_id),
                       category = EXCLUDED.category, priority = EXCLUDED.priority, status = EXCLUDED.status,
                       subject = EXCLUDED.subject, assigned_team = EXCLUDED.assigned_team,
                       resolution = COALESCE(EXCLUDED.resolution, support_tickets.resolution),
                       handoff_packet = COALESCE(EXCLUDED.handoff_packet, support_tickets.handoff_packet),
                       updated_at = EXCLUDED.updated_at
                   WHERE support_tickets.customer_id = EXCLUDED.customer_id
                   RETURNING *""",
                {"id": ticket_id, "customer": customer,
                 "order": order["id"] if order else None, "at": at, "category": category, "priority": priority,
                 "status": status, "subject": subject, "message": customer_message, "resolution": resolution,
                 "team": team, "packet": Jsonb(handoff_packet, dumps=json_dumps) if handoff_packet else None})
            if (row := cur.fetchone()) is None:
                raise WriteRefused(f"ticket {ticket_id} belongs to another customer")
            return row

    def replay(self, idempotency_key: str) -> dict[str, Any] | None:
        """The stored request and result of an earlier write with this key."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT request, result FROM backend_writes WHERE idempotency_key = %s", (idempotency_key,))
            return cur.fetchone()

    def apply(self, request: dict[str, Any], *, idempotency_key: str, reason: str, at: datetime,
              payment_id: str | None = None, return_id: str | None = None,
              customer_id: str | None = None) -> dict[str, Any]:
        """Run `request` (ticket_id, operation, order_id, amount, items). Raises WriteRefused. With a
        signed-in `customer_id`, another customer's order is refused as not found."""
        operation = request["operation"]
        if operation not in RECORD_PREFIXES:
            raise WriteRefused(f"unsupported write operation: {operation}")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM orders WHERE lower(id) = lower(%s) FOR UPDATE", (request["order_id"],))
            order = cur.fetchone()
            if order is None or (customer_id and order["customer_id"] != customer_id):
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
