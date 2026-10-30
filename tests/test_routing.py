"""Shared eligibility checks run before the architecture's approval routing."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from support_ops.policy.facts import BackendFacts

from support_ops.policy.routing import RoutingRules, at_least, strictest

NOW = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
DEFAULT_RULES = Path(__file__).resolve().parents[1] / "src/support_ops/policy/rules/rules.json"
REFUND = {
    "payment_status": "captured", "currency": "SGD", "captured_amount": 200,
    "refunded_amount": 0, "pending_refund_amount": 0, "approved_refund_amount": 200,
    "original_payment_method": True, "request_accepted": True, "inspection_or_waiver": True,
}
RETURN = {"delivered_at": NOW - timedelta(days=5), "requested_at": NOW,
          "reason": "faulty", "evidence_status": "verified"}


# Workflow tests supply verified inputs to exercise routing independently of missing DB fields.
# Production backend mapping (including unknown inspection/evidence facts) is tested separately.
def verified_rule_inputs():
    read = BackendFacts.read

    def collect(source, operation, order_id, item_ids=None, *, history=None):
        facts = read(source, operation, order_id, item_ids, history=history)
        if not facts:
            return facts
        if operation == "refund":
            verified = {"request_accepted": True, "original_payment_method": True,
                        "approved_refund_amount": facts["captured_amount"],
                        "inspection_or_waiver": bool(facts.get("delivered_at"))}
        elif operation in {"return_request", "exchange"}:
            verified = {"reason": "faulty", "evidence_status": "verified",
                        "alternative_resolution_agreed": True}
        elif operation == "cancel":
            verified = {"authority_verified": True, "label_created": False, "packing_started": False,
                        "any_quantity_fulfilled": False, "carrier_collected": False}
        else:
            verified = {}
        return {**verified, **facts}

    return patch.object(BackendFacts, "read", autospec=True, side_effect=collect)


class RouteOrderTests(unittest.TestCase):
    def test_routes_order_from_auto_to_human(self):
        self.assertTrue(at_least("human", "approver"))
        self.assertFalse(at_least("auto", "approver"))
        self.assertEqual(strictest("auto", "human", "approver"), "human")
        with self.assertRaises(ValueError):
            at_least("unknown", "auto")


class MinRouteTests(unittest.TestCase):
    def setUp(self):
        self.rules = RoutingRules.load()

    def test_refund_tiers_apply_only_after_verified_eligibility(self):
        for amount, route, check in ((30, "auto", "auto_approval"),
                                     (30.01, "approver", "automated_review"),
                                     (100, "approver", "automated_review"),
                                     (100.01, "human", "limit_exceeded")):
            with self.subTest(amount=amount):
                result = self.rules.min_route("refund", amount, business_facts=REFUND)
                self.assertEqual(result["route"], route)
                self.assertEqual(result["requires_human"], route == "human")
                self.assertEqual(result["policy_check"]["check"], f"refund.{check}")
                expected = self.rules.rules.data["refund"]["checks"][check]["outcome"]
                for field in ("allowed", "requires_human", "route", "reason"):
                    self.assertEqual(result["policy_check"][field], expected[field])
        missing = self.rules.min_route("refund", 5)
        self.assertFalse(missing["allowed"])
        self.assertEqual(missing["route"], "human")

    def test_human_conditions_override_a_small_refund(self):
        for facts in ({"disputed": True}, {"unsafe_item": True}, {"evidence_status": "unavailable"},
                      {"human_requested": True}):
            with self.subTest(facts=facts):
                result = self.rules.min_route("refund", 5, business_facts={**REFUND, **facts})
                self.assertEqual((result["allowed"], result["requires_human"], result["route"]),
                                 (False, True, "human"))

    def test_an_excessive_amount_is_blocked_instead_of_approved_by_qa(self):
        result = self.rules.min_route("refund", 20, business_facts={**REFUND, "refunded_amount": 190})
        self.assertEqual((result["allowed"], result["requires_human"], result["route"]), (False, False, None))
        result = self.rules.min_route("refund", 20, business_facts=REFUND, goods_value=5)
        self.assertFalse(result["allowed"])

    def test_return_authorization_does_not_use_the_refund_amount_tiers(self):
        result = self.rules.min_route("return", business_facts=RETURN)
        self.assertEqual((result["allowed"], result["route"]), (True, "auto"))
        late = self.rules.min_route("return", business_facts={**RETURN, "delivered_at": NOW - timedelta(days=31)})
        self.assertEqual(late["route"], "human")
        existing = self.rules.min_route("return", business_facts={**RETURN, "existing_return": True})
        self.assertEqual((existing["allowed"], existing["route"]), (False, None))

    def test_known_final_sale_change_of_mind_is_blocked(self):
        facts = {**RETURN, "reason": "change_of_mind", "final_sale": True, "unused": True,
                 "original_packaging": True, "tags_complete": True, "accessories_complete": True}
        result = self.rules.min_route("return", business_facts=facts)
        self.assertEqual((result["allowed"], result["requires_human"]), (False, False))

    def test_pending_status_alone_does_not_require_a_human(self):
        result = self.rules.min_route("none", pending_return=True)
        self.assertEqual((result["route"], result["pending_return"]), ("auto", True))

    def test_delays_use_the_policy_calendar_day_threshold(self):
        for days, route in ((1, "auto"), (3, "auto"), (4, "human")):
            with self.subTest(days=days):
                parcel = {"delivery_status": "in_transit", "estimated_delivery": NOW - timedelta(days=days), "as_of": NOW}
                result = self.rules.min_route("none", delivery_facts=[parcel], overdue_shipment=True)
                self.assertEqual(result["route"], route)
        lost = self.rules.min_route("none", delivery_facts=[{"delivery_status": "lost"}])
        self.assertEqual(lost["route"], "human")

    def test_cancellation_uses_verified_fulfillment_and_authority(self):
        facts = {"order_status": "processing", "authority_verified": True, "label_created": False,
                 "packing_started": False, "any_quantity_fulfilled": False, "carrier_collected": False}
        self.assertEqual(self.rules.min_route("order_change", change_type="cancel", business_facts=facts)["route"], "auto")
        self.assertEqual(self.rules.min_route("order_change", change_type="cancel", business_facts={**facts, "packing_started": True})["route"], "human")
        self.assertEqual(self.rules.min_route("order_change", change_type="address_change")["route"], "human")

    def test_alternative_resolution_requires_a_recorded_agreement(self):
        self.assertEqual(self.rules.min_route("exchange", business_facts=RETURN)["route"], "human")
        agreed = self.rules.min_route("exchange", business_facts={**RETURN, "alternative_resolution_agreed": True})
        self.assertEqual(agreed["route"], "approver")

    def test_unsupported_operations_require_human_review(self):
        self.assertEqual(self.rules.min_route("teleport")["route"], "human")


class LoadTests(unittest.TestCase):
    def test_limits_are_read_once_from_the_shared_business_rules(self):
        data = json.loads(DEFAULT_RULES.read_text())
        data["groups"]["refund"].update(max_auto_amount="15", max_reviewed_amount="80")
        data["groups"]["return_request"]["window_calendar_days"] = 14
        self.assertNotIn("refund_auto_max_amount", data["routing"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            path.write_text(json.dumps(data))
            rules = RoutingRules.load(path)
        self.assertEqual((rules.refund_auto_max, rules.refund_approver_max, rules.return_window_days), (15, 80, 14))

    def test_unknown_or_missing_operations_are_rejected(self):
        data = json.loads(DEFAULT_RULES.read_text())
        for operations in ({"teleport": "auto"}, {**data["routing"]["operations"], "refund": "invalid"}):
            with self.subTest(operations=operations), tempfile.TemporaryDirectory() as directory:
                data["routing"]["operations"] = operations
                path = Path(directory) / "rules.json"
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    RoutingRules.load(path)
