"""Collect existing backend records into temporary inputs for the rule engine."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from support_ops.backend.postgres import PostgresReader
from support_ops.helpers.clock import Clock

VOID_REFUND_STATUSES = {"failed", "cancelled", "rejected"}
VOID_RETURN_STATUSES = {"cancelled", "rejected", "closed"}


class BackendFacts:
    def __init__(self, reader: PostgresReader, clock: Clock):
        self.reader, self.clock = reader, clock

    # Map known backend fields only; leave unsupported inspection and approval facts absent.
    def read(self, operation: str, order_id: str, item_ids: list[str] | None = None,
             *, history: dict | None = None) -> dict[str, Any]:
        # Step 1: Establish the order and the affected items.
        order = self.reader.read("orders", order_id)
        if not isinstance(order, dict) or order.get("not_found"):
            return {}
        if operation not in {"cancel", "refund", "return_request", "exchange"}:
            return {}
        items = self.reader.items_for_order(order_id)
        order_items = {item["id"] for item in items}
        selected = set(item_ids or order_items)
        if not selected or not selected <= order_items:
            return {}
        shipments = self.reader.shipments_for_order(order_id)
        facts: dict[str, Any] = {}

        # Step 2: Map observed fulfillment and delivery facts without assuming missing flags are false.
        if operation == "cancel":
            facts["order_status"] = order.get("status")
            if any(row.get("shipped_at") or row.get("delivered_at") for row in shipments):
                facts["carrier_collected"] = True
            if any(row.get("status") == "label_created" for row in shipments):
                facts["label_created"] = True
            if any(item.get("item_status") == "fulfilled" for item in items):
                facts["any_quantity_fulfilled"] = True
            elif all(item.get("item_status") in {"open", "cancelled"} for item in items):
                facts["any_quantity_fulfilled"] = False
            return facts
        if len(shipments) == 1 and shipments[0].get("delivered_at"):
            facts["delivered_at"] = shipments[0]["delivered_at"]

        # Step 3: Read existing returns and refunds; an order-level return does not identify its items.
        history = history if history is not None else self.reader.returns_for_order(order_id)
        if operation in {"return_request", "exchange"}:
            facts["requested_at"] = self.clock.now().isoformat()
            active = [row for row in history.get("returns", []) if row.get("status") not in VOID_RETURN_STATUSES]
            facts["return_scope_unverified"] = bool(active)
            return facts

        # Step 4: Calculate refund balances from captured payments and existing refund records.
        payments = self.reader.payments_for_order(order_id)
        captured = [row for row in payments if row.get("status") == "captured"]
        refunds = history.get("refunds", [])
        facts.update(
            currency=order.get("currency"),
            payment_status="captured" if captured else "not_captured",
            captured_amount=sum((Decimal(str(row["amount"])) for row in captured), Decimal(0)),
            refunded_amount=sum(
                (Decimal(str(row["amount"])) for row in refunds if row.get("status") == "processed"), Decimal(0)),
            pending_refund_amount=sum(
                (Decimal(str(row["amount"])) for row in refunds
                 if row.get("status") not in VOID_REFUND_STATUSES | {"processed"}), Decimal(0)),
            goods_value=sum(
                (Decimal(str(item["unit_price"])) * item["quantity"] for item in items if item["id"] in selected),
                Decimal(0)),
        )
        if order.get("status") == "cancelled" and selected == order_items:
            facts["goods_value"] = Decimal(str(order["total"]))
        return facts

    # Check each parcel independently; one delivered parcel does not settle another's delay.
    def deliveries(self, order_id: str) -> list[dict[str, Any]]:
        return [{"delivery_status": row.get("status"), "estimated_delivery": row.get("estimated_delivery"),
                 "as_of": self.clock.now().isoformat()} for row in self.reader.shipments_for_order(order_id)]
