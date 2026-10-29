"""`route` tool: a specialist records its proposal and chosen route.

The facts the rule engine needs (order total, earlier refunds, delivery date) are read from
the backend here, never taken from the model. A route looser than the rule engine's minimum
is rejected with the minimum and its reasons, so the specialist can correct it.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from support_ops.agents.models import SPECIALIST_ACTIONS, ProposalDraft, Proposal, Specialist
from support_ops.agents.toolbox import ExtraTool
from support_ops.backend.postgres import PostgresReader
from support_ops.helpers.clock import Clock
from support_ops.policy.routing import WINDOWED, RoutingRules, at_least

# Refunds in these states never paid out, so they do not count against the order total.
VOID_REFUND_STATUSES = {"failed", "cancelled", "rejected"}

DESCRIPTION = (
    "Record your proposal and its route once the facts and policy are checked. "
    "auto: execute with no second check; approver: second check before execution; "
    "human: hand off to an operator. A route looser than the rules allow is rejected with "
    "the minimum. Use action none for information-only answers. This does not execute anything."
)


class RouteTool:
    def __init__(self, reader: PostgresReader, routing: RoutingRules, clock: Clock, specialist: Specialist):
        self.reader, self.routing, self.clock, self.specialist = reader, routing, clock, specialist

    def as_tool(self) -> ExtraTool:
        return ExtraTool(ProposalDraft, DESCRIPTION, self.run)

    def facts(self, proposal: Proposal) -> dict[str, Any]:
        if not proposal.order_id:
            return {}
        order = self.reader.read("orders", proposal.order_id)
        if order.get("not_found"):
            raise ValueError(f"order {proposal.order_id} not found")
        facts: dict[str, Any] = {}
        if proposal.action in {"refund", "return", "exchange"}:
            facts["order_total"] = order["total"]
        if proposal.items:
            facts["goods_value"] = self.goods_value(proposal)
        if proposal.action == "refund":
            refunds = self.reader.returns_for_order(proposal.order_id)["refunds"]
            facts["refunded_to_date"] = sum(
                (Decimal(str(refund["amount"])) for refund in refunds
                 if refund.get("status") not in VOID_REFUND_STATUSES),
                Decimal(0),
            )
        shipments = self.reader.shipments_for_order(proposal.order_id)
        if proposal.action in WINDOWED:
            delivered = [shipment["delivered_at"] for shipment in shipments if shipment.get("delivered_at")]
            facts["delivered_at"] = max(delivered) if delivered else None
        facts["overdue_shipment"] = any(self.overdue(shipment) for shipment in shipments)
        return facts

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
            value += Decimal(str(item["unit_price"])) * line.quantity
        return value

    def check(self, proposal: Proposal) -> dict[str, Any]:
        """The rule engine's minimum route for this proposal."""
        return self.routing.min_route(proposal.action, proposal.amount, now=self.clock.now(),
                                      policy_exception=proposal.policy_exception, **self.facts(proposal))

    def run(self, ticket_id: str, args: dict[str, Any]) -> dict[str, Any]:
        proposal = Proposal(specialist=self.specialist, **args)
        allowed = SPECIALIST_ACTIONS[self.specialist]
        if proposal.action not in allowed:
            raise ValueError(f"action {proposal.action!r} is outside the {self.specialist} specialist's domain "
                             f"(allowed: {', '.join(sorted(allowed))}); propose action none with route human "
                             "so the case is handed over")
        minimum = self.check(proposal)
        if not at_least(proposal.route, minimum["route"]):
            reasons = "; ".join(minimum["reasons"]) or "rule engine minimum"
            raise ValueError(f"route {proposal.route!r} rejected: the minimum is {minimum['route']!r} "
                             f"({reasons}); choose that route or a stricter one")
        return {"status": "recorded", "route": proposal.route,
                "proposal": proposal.model_dump(mode="json"), "min_route": minimum}
