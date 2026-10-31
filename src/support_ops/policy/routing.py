"""Apply shared policy checks before choosing a multi-agent approval route."""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from support_ops.helpers.config import ROOT
from support_ops.policy.rules import DEFAULT_RULES_FILE, Rules

Route = Literal["auto", "approver", "human"]
ROUTES: tuple[Route, ...] = ("auto", "approver", "human")
OPERATIONS = ("none", "refund", "return", "exchange", "order_change", "account_change")


# Compare routes from least to most restrictive.
def strictness(route: str) -> int:
    if route not in ROUTES:
        raise ValueError(f"unknown route: {route!r}; expected one of {', '.join(ROUTES)}")
    return ROUTES.index(route)


def at_least(route: str, minimum: str) -> bool:
    return strictness(route) >= strictness(minimum)


def strictest(*routes: str) -> Route:
    return max(routes, key=strictness)


@dataclass(frozen=True)
class RoutingRules:
    version: str
    rule_id: str
    currency: str
    refund_auto_max: Decimal
    refund_approver_max: Decimal
    return_window_days: int
    operations: dict[str, Route]
    rules: Rules
    awaiting_shop_return_statuses: tuple[str, ...]

    # Read workflow defaults; amount limits and eligibility stay in the shared checks.
    @classmethod
    def load(cls, path: str | Path = DEFAULT_RULES_FILE) -> RoutingRules:
        path = Path(path)
        resolved = path if path.is_absolute() else ROOT / path
        rules = Rules.load(resolved)
        try:
            routing = json.loads(resolved.read_text())["routing"]
            operations = routing["operations"]
            if set(operations) != set(OPERATIONS) or any(route not in ROUTES for route in operations.values()):
                raise ValueError("operations must name each supported action and a valid route")
            return cls(
                version=rules.version, rule_id=routing["rule_id"], currency=rules.currency,
                refund_auto_max=rules.refund_auto_limit, refund_approver_max=rules.refund_limit,
                return_window_days=rules.data["return_request"]["window_calendar_days"],
                operations=operations, rules=rules,
                awaiting_shop_return_statuses=tuple(routing["awaiting_shop_return_statuses"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid routing rules in {path}: {exc}") from exc

    # Business checks decide eligibility; workflow defaults may only make review stricter.
    def min_route(self, operation: str, amount: Any = None, *, business_facts: dict | None = None,
                  delivery_facts: list[dict] | None = None, change_type: str | None = None,
                  goods_value: Any = None, policy_exception: bool = False,
                  overdue_shipment: bool = False, pending_return: bool = False) -> dict[str, Any]:
        facts = dict(business_facts or {})
        if operation == "none":
            checks = [self.rules.check("delivery", facts=parcel) for parcel in delivery_facts or []]
            decision = next((check for check in checks if check["requires_human"]),
                            {"allowed": True, "requires_human": False, "route": "auto", "reason": "Information only."})
        elif operation == "refund":
            decision = self.rules.check("refund", amount, facts)
            if decision["allowed"] and goods_value is not None and Decimal(str(amount)) > Decimal(str(goods_value)):
                decision = {**decision, "allowed": False, "requires_human": False, "route": None,
                            "reason": "Refund exceeds what was paid for the listed items."}
        elif operation == "return":
            decision = self.rules.check("return_request", facts=facts)
        elif operation == "order_change" and change_type in {"cancel", "cancellation", "cancel_order"}:
            decision = self.rules.check("cancel", facts=facts)
        elif operation == "exchange" and facts.get("alternative_resolution_agreed") is True:
            decision = self.rules.check("return_request", facts=facts)
        else:
            decision = self.rules.check("unsupported_operation")

        route = decision["route"]
        if decision["allowed"]:
            route = strictest(route, self.operations[operation])
            if policy_exception:
                route = strictest(route, "approver")
        return {
            "operation": operation, "rule": self.rule_id, "version": self.version, "currency": self.currency,
            "allowed": decision["allowed"] and route != "human",
            "requires_human": decision["requires_human"] or route == "human", "route": route,
            "reasons": [decision["reason"]], "policy_check": decision,
            "overdue_shipment": overdue_shipment, "pending_return": pending_return,
        }
