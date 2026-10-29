"""Minimum route for a multi-agent proposal, from the `routing` section of policy/rules.yaml.

Routes run from loosest to strictest: auto < approver < human. A specialist may choose a
stricter route than the minimum, never a looser one. Facts passed in come from backend
reads, never from the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

import yaml

from support_ops.helpers.config import ROOT
from support_ops.policy.rules import DEFAULT_RULES_FILE, Rules

Route = Literal["auto", "approver", "human"]
ROUTES: tuple[Route, ...] = ("auto", "approver", "human")
OPERATIONS = ("none", "refund", "return", "exchange", "order_change", "account_change")
# Operations that act on a delivered item and so depend on the return window.
WINDOWED = {"refund", "return", "exchange"}


def strictness(route: str) -> int:
    if route not in ROUTES:
        raise ValueError(f"unknown route: {route!r}; expected one of {', '.join(ROUTES)}")
    return ROUTES.index(route)


def at_least(route: str, minimum: str) -> bool:
    """True when `route` is as strict as `minimum` or stricter."""
    return strictness(route) >= strictness(minimum)


def strictest(*routes: str) -> Route:
    return max(routes, key=strictness)


def _amount(value: Any, name: str) -> Decimal:
    if value is None:
        raise ValueError(f"{name} is required")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{name} must be finite and not negative")
    return amount


@dataclass(frozen=True)
class RoutingRules:
    version: str
    rule_id: str
    currency: str
    refund_auto_max: Decimal
    refund_approver_max: Decimal   # refund.max_auto_amount: the limit the single agent hands off above
    return_window_days: int
    operations: dict[str, Route]
    overdue_shipment_route: Route = "human"   # information-only answers about a late shipment

    @classmethod
    def load(cls, path: str | Path = DEFAULT_RULES_FILE) -> RoutingRules:
        path = Path(path)
        resolved = path if path.is_absolute() else ROOT / path
        rules = Rules.load(resolved)   # validates the shared section
        data = yaml.safe_load(resolved.read_text()) or {}
        try:
            routing = data["routing"]
            auto_max = Decimal(str(routing["refund_auto_max_amount"]))
            window = int(routing["return_window_days"])
            overdue = str(routing.get("overdue_shipment_route", "human"))
            operations = {str(name): str(route) for name, route in routing["operations"].items()}
            routing_rules = cls(rules.version, str(routing["rule_id"]), rules.currency,
                                auto_max, rules.refund_limit, window, operations, overdue)
        except (KeyError, TypeError, ValueError, AttributeError, InvalidOperation) as exc:
            raise ValueError(f"invalid routing rules in {path}: {exc!r}") from exc
        if not auto_max.is_finite() or not 0 < auto_max <= rules.refund_limit:
            raise ValueError(f"invalid routing rules in {path}: need 0 < refund_auto_max_amount "
                             "<= refund.max_auto_amount")
        if window < 0:
            raise ValueError(f"invalid routing rules in {path}: return_window_days must not be negative")
        if unknown := sorted(set(operations) - set(OPERATIONS)):
            raise ValueError(f"invalid routing rules in {path}: unknown operations {unknown}")
        if routing_rules.overdue_shipment_route not in ROUTES:
            raise ValueError(f"invalid routing rules in {path}: unknown overdue_shipment_route")
        if bad := sorted(name for name, route in operations.items() if route not in ROUTES):
            raise ValueError(f"invalid routing rules in {path}: unknown routes for {bad}")
        return routing_rules

    def min_route(self, operation: str, amount: Any = None, *, order_total: Any = None,
                  refunded_to_date: Any = 0, goods_value: Any = None, delivered_at: datetime | None = None,
                  now: datetime | None = None, policy_exception: bool = False,
                  overdue_shipment: bool = False) -> dict[str, Any]:
        """The loosest route this proposal may take. `allowed: False` means the backend
        will refuse the write whatever the route, so the case goes to a human."""
        result: dict[str, Any] = {
            "operation": operation, "rule": self.rule_id, "version": self.version,
            "currency": self.currency, "reasons": [],
        }
        if operation not in self.operations:
            return {**result, "route": "human", "allowed": False,
                    "reasons": [f"operation {operation!r} is not covered by the routing rules"]}
        route, allowed, reasons = self.operations[operation], True, result["reasons"]

        def raise_to(minimum: Route, reason: str) -> None:
            nonlocal route
            if strictness(minimum) > strictness(route):
                route = minimum
            reasons.append(reason)

        if operation == "refund":
            value = _amount(amount, "refund amount")
            if value == 0:
                raise ValueError("refund amount must be positive")
            if value > self.refund_approver_max:
                raise_to("human", f"refund above {self.refund_approver_max} {self.currency}")
            elif value > self.refund_auto_max:
                raise_to("approver", f"refund above {self.refund_auto_max} {self.currency}")
            if order_total is not None:
                remaining = _amount(order_total, "order total") - _amount(refunded_to_date, "refunded to date")
                if value > remaining:
                    allowed = False
                    raise_to("human", "refund exceeds the order total less earlier refunds")
        elif operation in {"return", "exchange"} and (goods_value is not None or order_total is not None):
            # Goods coming back usually become a refund, so they use the refund value tiers. Without
            # the value of the lines coming back, the whole order is assumed.
            subject = "goods" if goods_value is not None else "an order"
            value = _amount(goods_value if goods_value is not None else order_total, f"{subject} value")
            if value > self.refund_approver_max:
                raise_to("human", f"{operation} of {subject} above {self.refund_approver_max} {self.currency}")
            elif value > self.refund_auto_max:
                raise_to("approver", f"{operation} of {subject} above {self.refund_auto_max} {self.currency}")

        if operation in WINDOWED:
            if delivered_at is None:
                # A refund on an undelivered (e.g. cancelled) order has no window to check.
                if operation != "refund":
                    raise_to("human", "no delivery date to check the return window against")
            elif now is not None and (now - delivered_at).days > self.return_window_days:
                raise_to("human", f"more than {self.return_window_days} days after delivery")

        if overdue_shipment and operation == "none":
            # Only when the answer is just "it is late": a proposed remedy (refund, cancellation...)
            # keeps its own tiers. The flag below still lets the reply acknowledge the delay.
            raise_to(self.overdue_shipment_route, "a shipment is past its estimated delivery date")
        if policy_exception:
            raise_to("approver", "the proposal relies on a policy exception")
        return {**result, "route": route, "allowed": allowed, "overdue_shipment": overdue_shipment}
