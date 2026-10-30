"""`route` tool: a specialist records its proposal and chosen route.

The facts the rule engine needs (order total, earlier refunds, delivery date) are read from
the backend here, never taken from the model. A route looser than the rule engine's minimum
is rejected with the minimum and its reasons, so the specialist can correct it.
"""
from __future__ import annotations

from datetime import date
import json
from decimal import Decimal
from typing import Any

from support_ops.agents.models import SPECIALIST_ACTIONS, ProposalDraft, Proposal, Specialist
from support_ops.agents.toolbox import ExtraTool
from support_ops.backend.postgres import PostgresReader
from support_ops.helpers.clock import Clock
from support_ops.policy.routing import ROUTES, RoutingRules, at_least
from support_ops.policy.facts import BackendFacts, goods_value

# Refunds in these states never paid out, so they do not count against the order total.
VOID_REFUND_STATUSES = {"failed", "cancelled", "rejected"}
VOID_RETURN_STATUSES = {"cancelled", "rejected"}

DESCRIPTION = (
    "Record your proposal and its route once the facts and policy are checked. "
    "auto: execute with no second check; approver: second check before execution; "
    "human: hand off to an operator. A route looser than the rules allow is rejected with "
    "the minimum. Use action none for information-only answers. This does not execute anything."
)


class RouteTool:
    def __init__(self, reader: PostgresReader, routing: RoutingRules, clock: Clock, specialist: Specialist,
                 returns_checked: set[str] | None = None, orders_seen: list[str] | None = None):
        self.reader, self.routing, self.clock, self.specialist = reader, routing, clock, specialist
        self.backend_facts = BackendFacts(reader, clock)
        # Orders the specialist's read tools were called with, shared with the specialist's run.
        self.orders_seen = orders_seen if orders_seen is not None else []
        # Orders whose existing returns and refunds the specialist has seen (via its lookup tool or a
        # rejection from here). Shared with that lookup tool for the specialist run.
        self.returns_checked = returns_checked if returns_checked is not None else set()

    def as_tool(self) -> ExtraTool:
        return ExtraTool(ProposalDraft, DESCRIPTION, self.run)

    def order_context(self, proposal: Proposal) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """The rule engine's facts for this proposal, and the order's existing returns and refunds."""
        if not proposal.order_id:
            return {}, []
        order = self.reader.read("orders", proposal.order_id)
        if order.get("not_found"):
            if proposal.action == "none":
                # Answering about an order that does not exist, e.g. asking the customer to confirm
                # the number, is legitimate; there are simply no facts to check.
                return {}, []
            raise ValueError(f"order {proposal.order_id} not found: ask the customer to confirm the order "
                             "number (action none, route auto, with customer_question)")
        facts: dict[str, Any] = {}
        if proposal.items:
            facts["goods_value"] = self.goods_value(proposal)
        history = self.reader.returns_for_order(proposal.order_id)
        facts["history"] = history
        shipments = self.reader.shipments_for_order(proposal.order_id)
        delivered = [shipment["delivered_at"] for shipment in shipments if shipment.get("delivered_at")]
        delivered_at = max(delivered) if delivered else None
        returns = [row for row in history.get("returns", []) if row.get("status") not in VOID_RETURN_STATUSES]
        facts["pending_return"] = any(row.get("status") in self.routing.awaiting_shop_return_statuses
                                      for row in returns)
        facts["overdue_shipment"] = any(self.overdue(shipment) for shipment in shipments)
        return facts, existing_requests(history, delivered_at, self.routing.return_window_days,
                                        self.routing.awaiting_shop_return_statuses)

    def overdue(self, shipment: dict[str, Any]) -> bool:
        """Not delivered, not cancelled, and past its estimated delivery date."""
        estimated = shipment.get("estimated_delivery")
        if shipment.get("delivered_at") or not estimated or shipment.get("status") == "cancelled":
            return False
        estimated = estimated if isinstance(estimated, date) else date.fromisoformat(str(estimated)[:10])
        return estimated < self.clock.now().date()

    def goods_value(self, proposal: Proposal) -> Decimal:
        """Value of the order lines coming back, priced from the order itself, not the model."""
        lines = {item["id"]: item for item in self.reader.items_for_order(proposal.order_id)}
        value = Decimal(0)
        for line in proposal.items:
            item = lines.get(line.item_id)
            if item is None:
                raise ValueError(f"item {line.item_id} is not in order {proposal.order_id}; "
                                 f"its items are {sorted(lines)}")
            if line.quantity > item["quantity"]:
                raise ValueError(f"item {line.item_id} has quantity {item['quantity']}, not {line.quantity}")
            value += goods_value(item, line.quantity)
        return value

    def check(self, proposal: Proposal) -> dict[str, Any]:
        """The rule engine's minimum route for this proposal, with the order's existing requests."""
        orders = sorted(set(self.orders_seen))
        if proposal.action == "none" and not proposal.order_id and len(orders) > 1:
            return self.check_each(proposal, orders)
        facts, existing = self.order_context(proposal)
        operation = {"return": "return_request", "order_change": "cancel"}.get(proposal.action, proposal.action)
        recorded = self.backend_facts.read(operation, proposal.order_id, [line.item_id for line in proposal.items],
                                          history=facts.get("history")) \
            if proposal.order_id and proposal.action != "none" else {}
        parcels = self.backend_facts.deliveries(proposal.order_id) \
            if proposal.order_id and proposal.action == "none" else []
        minimum = self.routing.min_route(
            proposal.action, proposal.amount, business_facts=recorded, delivery_facts=parcels,
            change_type=proposal.reason_code, goods_value=facts.get("goods_value"),
            policy_exception=proposal.policy_exception,
            overdue_shipment=facts.get("overdue_shipment", False), pending_return=facts.get("pending_return", False),
        )
        return {**minimum, "existing_requests": existing} if existing else minimum

    def check_each(self, proposal: Proposal, orders: list[str]) -> dict[str, Any]:
        """An answer about several orders: the strictest of their minimums, so a late shipment or a
        investigation condition on any of them still counts."""
        results = {order: self.check(proposal.model_copy(update={"order_id": order})) for order in orders}
        strictest = max(results.values(), key=lambda minimum: ROUTES.index(minimum["route"]))
        merged = {**strictest, "orders": orders,
                  "reasons": [f"{order}: {reason}" for order, minimum in results.items() for reason in minimum["reasons"]],
                  "allowed": all(minimum["allowed"] for minimum in results.values())}
        for key in ("overdue_shipment", "pending_return"):
            if any(key in minimum for minimum in results.values()):
                merged[key] = any(minimum.get(key) for minimum in results.values())
        existing = [request for minimum in results.values() for request in minimum.get("existing_requests", [])]
        if existing:
            merged["existing_requests"] = existing
        return merged

    def run(self, ticket_id: str, args: dict[str, Any]) -> dict[str, Any]:
        proposal = Proposal(specialist=self.specialist, **args)
        allowed = SPECIALIST_ACTIONS[self.specialist]
        if proposal.action not in allowed:
            raise ValueError(f"action {proposal.action!r} is outside the {self.specialist} specialist's domain "
                             f"(allowed: {', '.join(sorted(allowed))}); propose action none with route human "
                             "so the case is handed over")
        if proposal.action == "refund" and proposal.items and proposal.amount is not None:
            paid = self.goods_value(proposal)
            if Decimal(str(proposal.amount)) > paid:
                raise ValueError(f"refund {proposal.amount:.2f} exceeds the {paid:.2f} paid for the listed items; "
                                 "refund at most what was paid for them")
        minimum = self.check(proposal)
        existing = minimum.get("existing_requests", [])
        unseen = [row for row in existing if not (row["kind"] == "return" and row["status"] == "approved")]
        if unseen and proposal.order_id \
                and proposal.order_id not in self.returns_checked:
            # Deciding without knowing about a pending return or an earlier refund gets eligibility
            # wrong, so the specialist sees them once before its proposal is recorded.
            self.returns_checked.add(proposal.order_id)
            raise ValueError(f"order {proposal.order_id} already has returns or refunds on record: "
                             f"{json.dumps(existing, default=str)}. Take them into account (for example, "
                             "report a pending return's status rather than opening a duplicate, and count "
                             "earlier refunds), then call route again.")
        if not minimum["allowed"] and not minimum["requires_human"]:
            raise ValueError(f"proposed action is blocked: {'; '.join(minimum['reasons'])}; "
                             "correct the proposal or explain the policy using action none")
        if not at_least(proposal.route, minimum["route"]):
            reasons = "; ".join(minimum["reasons"]) or "rule engine minimum"
            raise ValueError(f"route {proposal.route!r} rejected: the minimum is {minimum['route']!r} "
                             f"({reasons}); choose that route or a stricter one")
        return {"status": "recorded", "route": proposal.route,
                "proposal": proposal.model_dump(mode="json"), "min_route": minimum}


def existing_requests(history: dict[str, list[dict[str, Any]]], delivered_at: Any = None,
                      window_days: int | None = None, awaiting_shop: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """An order's returns and refunds that still count, oldest first. A return says whether it was
    requested within the return window and whether the store still has to act on it."""
    def day(value: Any) -> str | None:
        return str(value)[:10] if value else None

    def in_window(row: dict[str, Any]) -> bool | None:
        requested = row.get("requested_at")
        if not (requested and delivered_at and window_days is not None):
            return None
        return (requested - delivered_at).days <= window_days

    returns = [{"kind": "return", "id": row.get("id"), "status": row["status"], "reason": row.get("reason"),
                "requested": day(row.get("requested_at")), "received": day(row.get("received_at")),
                "requested_within_return_window": in_window(row),
                "awaiting_the_store": row["status"] in awaiting_shop}
               for row in history.get("returns", []) if row.get("status") not in VOID_RETURN_STATUSES]
    refunds = [{"kind": "refund", "id": row.get("id"), "status": row["status"], "amount": str(row.get("amount")),
                "requested": day(row.get("requested_at"))}
               for row in history.get("refunds", []) if row.get("status") not in VOID_REFUND_STATUSES]
    return sorted(returns + refunds, key=lambda request: request["requested"] or "")
