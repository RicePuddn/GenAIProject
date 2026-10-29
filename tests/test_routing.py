"""Minimum-route rules for the multi-agent arm (policy/routing.py)."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from support_ops.policy.routing import RoutingRules, at_least, strictest

NOW = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)

RULES_YAML = """
version: "test"
currency: SGD
refund:
  rule_id: test_limit
  max_auto_amount: "100.00"
routing:
  rule_id: test_routes
  refund_auto_max_amount: "{auto_max}"
  return_window_days: {window}
  operations:
    none: auto
    refund: auto
    return: auto
    {extra}
"""


def write_rules(directory: str, auto_max="30.00", window="30", extra="") -> Path:
    path = Path(directory) / "rules.yaml"
    path.write_text(RULES_YAML.format(auto_max=auto_max, window=window, extra=extra))
    return path


class RouteOrderTests(unittest.TestCase):
    def test_routes_order_from_auto_to_human(self):
        self.assertTrue(at_least("human", "approver"))
        self.assertTrue(at_least("approver", "approver"))
        self.assertFalse(at_least("auto", "approver"))
        self.assertEqual(strictest("auto", "human", "approver"), "human")
        with self.assertRaises(ValueError):
            at_least("whenever", "auto")


class MinRouteTests(unittest.TestCase):
    def setUp(self):
        self.rules = RoutingRules.load()

    def route(self, *args, **kwargs):
        if args and args[0] == "refund":
            kwargs.setdefault("goods_value", 10_000)   # a refund for items worth more than it, unless a test says otherwise
        return self.rules.min_route(*args, **kwargs)["route"]

    def test_repository_rules_share_the_single_agent_human_limit(self):
        self.assertEqual(self.rules.refund_approver_max, 100)
        self.assertEqual(self.rules.refund_auto_max, 30)
        self.assertEqual(self.rules.version, "2026-10-30")

    def test_refund_amount_tiers_include_their_upper_bounds(self):
        for amount, expected in ((30, "auto"), (30.01, "approver"), (100, "approver"), (100.01, "human")):
            with self.subTest(amount=amount):
                self.assertEqual(self.route("refund", amount), expected)

    def test_refund_above_what_is_left_of_the_order_is_refused(self):
        result = self.rules.min_route("refund", 20, order_total=50, refunded_to_date=40)
        self.assertEqual((result["route"], result["allowed"]), ("human", False))
        self.assertIn("exceeds the order total", result["reasons"][0])
        self.assertTrue(self.rules.min_route("refund", 10, order_total=50, refunded_to_date=40)["allowed"])

    def test_return_window_counts_whole_days_after_delivery(self):
        self.assertEqual(self.route("return", delivered_at=NOW - timedelta(days=30), now=NOW), "auto")
        self.assertEqual(self.route("return", delivered_at=NOW - timedelta(days=31), now=NOW), "human")
        self.assertEqual(self.route("refund", 10, delivered_at=NOW - timedelta(days=31), now=NOW), "human")

    def test_returns_and_exchanges_use_the_refund_value_tiers_on_the_order_total(self):
        delivered = {"delivered_at": NOW - timedelta(days=3), "now": NOW}
        for operation in ("return", "exchange"):
            with self.subTest(operation):
                self.assertEqual(self.route(operation, order_total=189, **delivered), "human")
                self.assertEqual(self.route(operation, order_total=60, **delivered), "approver")
        self.assertEqual(self.route("return", order_total=25, **delivered), "auto")

    def test_return_without_a_delivery_date_needs_a_human_but_a_refund_does_not(self):
        self.assertEqual(self.route("return"), "human")
        self.assertEqual(self.route("refund", 10), "auto")

    def test_overdue_shipment_needs_a_human_and_is_flagged(self):
        result = self.rules.min_route("none", overdue_shipment=True)
        self.assertEqual((result["route"], result["overdue_shipment"]), ("human", True))
        self.assertIn("past its estimated delivery date", result["reasons"][0])
        self.assertFalse(self.rules.min_route("none")["overdue_shipment"])

    def test_a_remedy_for_a_late_shipment_keeps_its_own_tiers(self):
        refund = self.rules.min_route("refund", 76.30, order_total=76.30, goods_value=76.30, overdue_shipment=True)
        self.assertEqual((refund["route"], refund["overdue_shipment"]), ("approver", True))
        self.assertEqual(self.route("refund", 20, overdue_shipment=True), "auto")
        self.assertEqual(self.route("refund", 209.28, overdue_shipment=True), "human")
        self.assertEqual(self.route("order_change", overdue_shipment=True), "approver")

    def test_a_refund_without_items_needs_the_approver(self):
        result = self.rules.min_route("refund", 10)
        self.assertEqual(result["route"], "approver")
        self.assertIn("names no order items", result["reasons"][0])

    def test_a_refund_above_what_the_items_cost_is_refused(self):
        # PR #3: $5 socks in a $100 order, customer claims $30.
        result = self.rules.min_route("refund", 30, order_total=100, goods_value=5)
        self.assertEqual((result["route"], result["allowed"]), ("human", False))
        self.assertTrue(self.rules.min_route("refund", 5, order_total=100, goods_value=5)["allowed"])

    def test_a_refund_on_a_shipped_undelivered_parcel_is_checked(self):
        # PR #3: a $25 order, shipped, not arrived.
        self.assertEqual(self.route("refund", 25, order_total=25, undelivered_shipment=True), "approver")
        self.assertEqual(self.route("refund", 25, order_total=25, undelivered_shipment=True,
                                    overdue_shipment=True), "human")
        # A cancelled order that never shipped keeps the plain tiers.
        self.assertEqual(self.route("refund", 25, order_total=25), "auto")

    def test_a_return_the_store_has_not_processed_needs_a_human_for_any_proposal(self):
        for operation in ("none", "refund", "return"):
            with self.subTest(operation):
                result = self.rules.min_route(operation, 10 if operation == "refund" else None, goods_value=10,
                                              delivered_at=NOW - timedelta(days=3), now=NOW, pending_return=True)
                self.assertEqual((result["route"], result["pending_return"]), ("human", True))
                self.assertIn("has not been processed by the store", " ".join(result["reasons"]))

    def test_the_window_is_measured_to_when_the_return_was_requested(self):
        delivered = NOW - timedelta(days=50)
        in_time = delivered + timedelta(days=4)   # asked 4 days after delivery; still unprocessed today
        self.assertEqual(self.route("refund", 10, delivered_at=delivered, now=NOW), "human")   # judged against today
        self.assertEqual(self.route("refund", 10, delivered_at=delivered, now=NOW, return_requested_at=in_time),
                         "auto")
        late = delivered + timedelta(days=31)
        self.assertEqual(self.route("refund", 10, delivered_at=delivered, now=NOW, return_requested_at=late), "human")

    def test_policy_exception_needs_at_least_the_approver(self):
        self.assertEqual(self.route("refund", 10, policy_exception=True), "approver")
        self.assertEqual(self.route("refund", 500, policy_exception=True), "human")

    def test_operation_baselines_and_unknown_operations(self):
        self.assertEqual(self.route("none"), "auto")
        self.assertEqual(self.route("order_change"), "approver")
        self.assertEqual(self.route("account_change"), "human")
        unknown = self.rules.min_route("teleport")
        self.assertEqual((unknown["route"], unknown["allowed"]), ("human", False))

    def test_bad_refund_amounts_raise(self):
        for amount in (None, 0, -1, "abc", float("nan"), float("inf")):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                self.rules.min_route("refund", amount)

    def test_result_records_the_rule_and_version(self):
        result = self.rules.min_route("refund", 50, goods_value=50)
        self.assertEqual((result["rule"], result["version"], result["currency"]),
                         ("mock_route_tiers_v1", "2026-10-30", "SGD"))
        self.assertEqual(result["reasons"], ["refund above 30.00 SGD"])


class LoadTests(unittest.TestCase):
    def test_invalid_routing_sections_are_rejected(self):
        cases = {
            "auto max above human limit": {"auto_max": "150"},
            "zero auto max": {"auto_max": "0"},
            "bad auto max": {"auto_max": "abc"},
            "negative window": {"window": "-1"},
            "unknown operation": {"extra": "teleport: auto"},
            "unknown route": {"extra": "exchange: whenever"},
            "unknown overdue route": {"extra": "none: auto\n  overdue_shipment_route: whenever"},
            "unknown pending return route": {"extra": "none: auto\n  pending_return_route: whenever"},
        }
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(RoutingRules.load(write_rules(directory)).rule_id, "test_routes")
            for name, overrides in cases.items():
                with self.subTest(name), self.assertRaises(ValueError):
                    RoutingRules.load(write_rules(directory, **overrides))
            path = Path(directory) / "rules.yaml"
            path.write_text('version: x\ncurrency: SGD\nrefund: {rule_id: r, max_auto_amount: "100"}\n')
            with self.assertRaises(ValueError):
                RoutingRules.load(path)


if __name__ == "__main__":
    unittest.main()
