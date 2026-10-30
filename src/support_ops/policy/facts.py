"""Collect operational records into temporary inputs for deterministic policy checks."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from support_ops.backend.postgres import PostgresReader
from support_ops.helpers.clock import Clock

VOID_REFUND_STATUSES = {"failed", "cancelled", "rejected"}
VOID_RETURN_STATUSES = {"cancelled", "rejected", "closed"}


# The paid line amount includes its recorded discount and tax allocation.
def goods_value(item: dict, quantity: int | None = None) -> Decimal:
    paid = item.get("paid_amount")
    total = Decimal(str(paid)) if paid is not None else Decimal(str(item["unit_price"])) * item["quantity"]
    selected_quantity = item["quantity"] if quantity is None else quantity
    return (total * selected_quantity / item["quantity"]).quantize(Decimal("0.01"))


class BackendFacts:
    def __init__(self, reader: PostgresReader, clock: Clock):
        self.reader, self.clock = reader, clock

    # Read the affected goods, then collect only the fields needed by the operation.
    def read(self, operation: str, order_id: str, item_ids: list[str] | None = None,
             *, history: dict | None = None) -> dict[str, Any]:
        order = self.reader.read("orders", order_id)
        if not isinstance(order, dict) or order.get("not_found"):
            return {}
        if operation not in {"cancel", "refund", "return_request", "return_dispatch", "exchange"}:
            return {}
        items = self.reader.items_for_order(order_id)
        selected = set(item_ids or [item["id"] for item in items])
        if not selected or not selected <= {item["id"] for item in items}:
            return {"order_status": order.get("status")} if operation == "cancel" else {}
        affected = [item for item in items if item["id"] in selected]
        shipments = self.reader.shipments_for_order(order_id)
        if operation == "cancel":
            return self._cancellation(order, items, shipments)

        # Resolve delivery using parcel-to-item links. Ambiguous historical split parcels stay unknown.
        parcels = [row for row in shipments if selected.intersection(
            line["order_item_id"] for line in row.get("items", []))]
        if not parcels and len(shipments) == 1:
            parcels = shipments
        covered = {line["order_item_id"] for row in parcels for line in row.get("items", [])}
        facts: dict[str, Any] = {}
        if parcels and all(row.get("delivered_at") for row in parcels) and (selected <= covered or len(shipments) == 1):
            facts["delivered_at"] = min(row["delivered_at"] for row in parcels)
        types = {item.get("product_type") for item in affected}
        if None not in types:
            facts["item_type"] = next((kind for kind in types if kind != "physical"), "physical")
        history = history if history is not None else self.reader.returns_for_order(order_id)
        facts.update(self._return_review(history, selected, affected, operation))
        if operation in {"return_request", "exchange"}:
            facts.setdefault("requested_at", self.clock.now().isoformat())
        if operation == "refund":
            if not facts.get("delivered_at") and facts.get("inspection_status") != "waived":
                facts["inspection_or_waiver"] = False
            whole_order = selected == {item["id"] for item in items}
            facts.update(self._payment(order, affected, history.get("refunds", []), whole_order))
        return facts

    # Order-item states describe packing; shipment records describe label creation and handover.
    def _cancellation(self, order: dict, items: list[dict], shipments: list[dict]) -> dict:
        facts = {"order_status": order.get("status")}
        states = {item.get("item_status") for item in items}
        if states <= {"open", "packing", "packed", "fulfilled", "cancelled"}:
            facts["packing_started"] = bool(states.intersection({"packing", "packed", "fulfilled"}))
            facts["any_quantity_fulfilled"] = "fulfilled" in states
        known = {"processing", "label_created", "shipped", "in_transit", "delayed", "delivery_attempted",
                 "out_for_delivery", "delivered", "cancelled"}
        if all(row.get("status") in known for row in shipments):
            facts["label_created"] = any(row.get("label_created_at") or row.get("status") == "label_created"
                                         for row in shipments)
            facts["carrier_collected"] = any(row.get("shipped_at") or row.get("delivered_at") or row.get("status")
                in {"shipped", "in_transit", "delayed", "delivery_attempted", "out_for_delivery", "delivered"}
                for row in shipments)
        return facts

    # Use one matching return review; never apply a different item's findings to the selected goods.
    def _return_review(self, history: dict, selected: set[str], items: list[dict], operation: str) -> dict:
        active = [row for row in history.get("returns", []) if row.get("status") not in VOID_RETURN_STATUSES]
        matches = [row for row in active if selected <= {line["order_item_id"] for line in row.get("items", [])}]
        if len(matches) != 1:
            return {"return_scope_unverified": bool(active)}
        request = matches[0]
        lines = [line for line in request["items"] if line["order_item_id"] in selected]
        accepted = request.get("status") == "approved" and bool(request.get("approved_at"))
        facts = {
            "return_id": request["id"], "reason": request.get("reason"),
            "requested_at": request.get("requested_at"), "request_accepted": accepted,
            "disputed": request.get("dispute_status") == "open",
            "alternative_resolution_agreed": accepted and request.get("resolution") == "exchange",
        }
        evidence = request.get("evidence_status")
        if evidence is not None:
            if evidence == "verified" and not request.get("reviewed_at"):
                evidence = "pending"
            facts["evidence_status"] = evidence
        if request.get("inspection_status") is not None:
            facts["inspection_status"] = request["inspection_status"]
            facts["inspection_or_waiver"] = bool(request.get("reviewed_at")) and request["inspection_status"] in {"passed", "waived"}
        # Completed and pending payments consume this return's approval as well as the order balance.
        refunds = [row for row in history.get("refunds", []) if row.get("return_id") == request["id"]
                   and row.get("status") not in VOID_REFUND_STATUSES] if operation == "refund" else []
        full_return = selected == {line["order_item_id"] for line in request["items"]}
        # An earlier refund has no per-item allocation; leave a partial return's remaining amount unknown.
        if (not refunds or full_return) and all(line.get("approved_refund_amount") is not None for line in lines):
            approved = sum((Decimal(str(line["approved_refund_amount"])) for line in lines), Decimal(0))
            used = sum((Decimal(str(row["amount"])) for row in refunds), Decimal(0))
            facts["approved_refund_amount"] = max(Decimal(0), approved - used)
        for name in ("unused", "original_packaging", "tags_complete", "accessories_complete"):
            if all(type(line.get(name)) is bool for line in lines):
                facts[name] = all(line[name] for line in lines)
        if all(type(item.get("final_sale")) is bool for item in items):
            facts["final_sale"] = any(item["final_sale"] for item in items)
        if operation == "return_request":
            facts["existing_return"] = request.get("status") in {"approved", "in_transit", "received"}
        for name in ("authorized_at", "handed_over_at", "extension_until"):
            if request.get(name):
                facts[name] = request[name]
        facts["extension_approved"] = bool(request.get("extension_approved_at"))
        facts["carrier_accepted"] = bool(request.get("handed_over_at"))
        return facts

    # Derive balances from original captured payments; callers cannot select a different destination.
    def _payment(self, order: dict, items: list[dict], refunds: list[dict], whole_order: bool) -> dict:
        payments = self.reader.payments_for_order(order["id"])
        captured = [row for row in payments if row.get("status") == "captured"]
        value = sum((goods_value(item) for item in items), Decimal(0))
        if order.get("status") == "cancelled" and whole_order:
            value = Decimal(str(order["total"]))
        return {
            "currency": order.get("currency"), "payment_status": "captured" if captured else "not_captured",
            "payment_id": captured[0]["id"] if len(captured) == 1 else None,
            "captured_amount": sum((Decimal(str(row["amount"])) for row in captured), Decimal(0)),
            "refunded_amount": sum((Decimal(str(row["amount"])) for row in refunds if row.get("status") == "processed"), Decimal(0)),
            "pending_refund_amount": sum((Decimal(str(row["amount"])) for row in refunds
                 if row.get("status") not in VOID_REFUND_STATUSES | {"processed"}), Decimal(0)),
            "goods_value": value,
        }

    # Check every parcel independently; one delivered parcel does not settle another's delay.
    def deliveries(self, order_id: str) -> list[dict[str, Any]]:
        return [{"delivery_status": row.get("status"), "estimated_delivery": row.get("estimated_delivery"),
                 "as_of": self.clock.now().isoformat()} for row in self.reader.shipments_for_order(order_id)]
