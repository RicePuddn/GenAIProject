"""Recorded facts feed the rules; facts the records do not hold stay unknown, and the write gate holds to them."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from support_ops.shared.clock import Clock
from support_ops.shared.db import PostgresReader
from support_ops.shared.rules import BackendFacts, Rules
from tests.fakes import FakeDatabase, services

NOW = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
ORDER = "ORD-2026-00001"
REFUND = {
    "payment_id": "PAY-1", "payment_status": "captured", "currency": "SGD", "captured_amount": 200,
    "refunded_amount": 0, "pending_refund_amount": 0, "approved_refund_amount": 200,
    "request_accepted": True, "inspection_or_waiver": True,
}


def reader(total=80, refunds=(), delivered_days_ago=5):
    """One order: two lines, one shipment, a captured payment and an approved return of both lines."""
    mock = Mock(spec=PostgresReader)
    mock.read.side_effect = lambda kind, record_id: (
        {"id": record_id, "total": total, "currency": "SGD", "status": "processing"} if record_id == ORDER
        else {"not_found": True, "kind": kind, "id": record_id})
    mock.returns_for_order.return_value = {"returns": [], "refunds": list(refunds)}
    delivered = None if delivered_days_ago is None else NOW - timedelta(days=delivered_days_ago)
    mock.shipments_for_order.return_value = [{"id": "SHP-1", "status": "delivered" if delivered_days_ago is not None
                                              else "in_transit", "delivered_at": delivered}]
    mock.items_for_order.return_value = [
        {"id": "ITM-1", "name": "Headphones", "quantity": 1, "unit_price": "189.00"},
        {"id": "ITM-2", "name": "Bottle", "quantity": 2, "unit_price": "32.00"},
    ]
    mock.payments_for_order.return_value = [{"id": "PAY-1", "amount": total, "status": "captured"}]
    for item in mock.items_for_order.return_value:
        item.update(final_sale=False, paid_amount=str(float(item["unit_price"]) * item["quantity"]),
                    product_type="physical", item_status="fulfilled")
    approved = {"id": "RET-APPROVED", "status": "approved", "reason": "faulty",
                "requested_at": NOW - timedelta(days=4), "received_at": NOW - timedelta(days=3),
                "approved_at": NOW - timedelta(days=2), "reviewed_at": NOW - timedelta(days=2),
                "evidence_status": "verified", "inspection_status": "passed", "dispute_status": "none",
                "items": [{"order_item_id": item["id"], "quantity": item["quantity"], "unused": True,
                           "original_packaging": True, "tags_complete": True, "accessories_complete": True,
                           "approved_refund_amount": min(total, float(item["paid_amount"]))}
                          for item in mock.items_for_order.return_value]}
    if delivered_days_ago is not None:
        mock.returns_for_order.return_value["returns"] = [approved]
    return mock


class BackendFactTests(unittest.TestCase):
    def tools(self, backend=None, **kwargs):
        return services(backend or reader(), writer=FakeDatabase(), **kwargs)

    @staticmethod
    def write(tools, amount=20, **details):
        """A refund of ITM-1; unless the test names its citations, it reads and cites the refunds policy."""
        if "cited_clauses" not in details:
            tools.call("policy", action="read", skill="returns-and-refunds", file="refunds.md")
            details["cited_clauses"] = ["refunds.md"]
        return tools.call("backend_write", **{"operation": "refund", "order_id": ORDER, "reason": "damaged_item",
                                              "items": [{"item_id": "ITM-1", "quantity": 1}], "amount": amount,
                                              **details})

    def test_actual_records_provide_balances_without_implying_inspection_or_approval(self):
        backend = reader(total=80, refunds=[
            {"amount": "10", "status": "processed"}, {"amount": "5", "status": "pending"},
            {"amount": "7", "status": "failed"},
        ])
        backend.returns_for_order.return_value["returns"] = [
            {"status": "received", "received_at": NOW, "reason": "damaged on arrival"},
        ]
        facts = BackendFacts(backend, Clock(NOW)).read("refund", ORDER, ["ITM-1"])
        self.assertEqual((facts["captured_amount"], facts["refunded_amount"], facts["pending_refund_amount"]), (80, 10, 5))
        self.assertEqual(facts["goods_value"], 189)
        for name in ("inspection_or_waiver", "request_accepted", "approved_refund_amount"):
            self.assertNotIn(name, facts)

    def test_unrecorded_prerequisites_need_a_person_and_the_agent_cannot_supply_them(self):
        for mode in ("rag", "skills"):
            with self.subTest(mode=mode):
                backend = reader()
                backend.returns_for_order.return_value = {"returns": [], "refunds": []}
                tools = self.tools(backend, mode=mode)
                cite = "REF-Refunds-6.1" if mode == "rag" else "refunds.md"
                if mode == "rag":
                    tools.call("policy", action="search", query="refund")
                self.assertEqual(self.write(tools, cited_clauses=[cite])["status"], "awaiting_human")
                self.assertEqual(tools.executor.writer.writes, [])
                supplied = self.write(tools, cited_clauses=[cite], facts={"inspection_or_waiver": True})
                self.assertIn("Extra inputs are not permitted", supplied["error"])   # never read as facts
                refused = tools.call("policy", action="check", operation="refund", order_id=ORDER)
                self.assertIn("error", refused)   # the policy tool's check covers deliveries only

    def test_unknown_item_ids_cannot_use_another_items_records(self):
        result = self.write(self.tools(), items=[{"item_id": "OTHER-ITEM", "quantity": 1}])
        self.assertEqual(result["status"], "awaiting_human")

    def test_refund_cannot_reuse_an_approval_already_paid_or_pending(self):
        for status in ("processed", "pending"):
            with self.subTest(status=status):
                backend = reader(total=100, refunds=[{"return_id": "RET-APPROVED", "status": status, "amount": "20.00"}])
                request = backend.returns_for_order.return_value["returns"][0]
                request["items"] = [{**request["items"][0], "approved_refund_amount": "20.00"}]
                tools = self.tools(backend)
                result = self.write(tools)
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(result["policy_check"]["check"], "refund.approved_amount_exceeded")
                self.assertEqual(tools.executor.writer.writes, [])

    def test_remaining_approval_excludes_failed_and_other_return_refunds(self):
        backend = reader(total=100, refunds=[
            {"return_id": "RET-APPROVED", "status": "processed", "amount": "10.00"},
            {"return_id": "RET-APPROVED", "status": "failed", "amount": "20.00"},
            {"return_id": "RET-OTHER", "status": "processed", "amount": "5.00"},
        ])
        request = backend.returns_for_order.return_value["returns"][0]
        request["items"] = [{**request["items"][0], "approved_refund_amount": "30.00"}]
        self.assertEqual(BackendFacts(backend, Clock(NOW)).read("refund", ORDER, ["ITM-1"])["approved_refund_amount"], 20)
        self.assertEqual(self.write(self.tools(backend))["status"], "executed")

    def test_partial_refund_allocation_stays_unknown_for_a_multi_item_return(self):
        backend = reader(total=100, refunds=[{"return_id": "RET-APPROVED", "status": "processed", "amount": "20.00"}])
        tools = self.tools(backend)
        self.assertEqual(self.write(tools)["status"], "awaiting_human")
        self.assertEqual(tools.executor.writer.writes, [])

    def test_open_item_states_and_no_shipments_allow_pre_fulfillment_cancellation(self):
        backend = reader()
        backend.shipments_for_order.return_value = []
        for item in backend.items_for_order.return_value:
            item["item_status"] = "open"
        facts = BackendFacts(backend, Clock(NOW)).read("cancel", ORDER)
        self.assertEqual(facts, {"order_status": "processing", "any_quantity_fulfilled": False,
                                 "packing_started": False, "label_created": False, "carrier_collected": False})
        self.assertTrue(Rules.load().check("cancel", facts=facts)["allowed"])

    def test_fulfillment_events_are_mapped_when_the_records_establish_them(self):
        backend = reader()
        facts = BackendFacts(backend, Clock(NOW)).read("cancel", ORDER)
        self.assertTrue(facts["carrier_collected"])
        backend.shipments_for_order.return_value = [{"status": "label_created"}]
        backend.items_for_order.return_value[0]["item_status"] = "fulfilled"
        facts = BackendFacts(backend, Clock(NOW)).read("cancel", ORDER)
        self.assertTrue(facts["label_created"])
        self.assertTrue(facts["any_quantity_fulfilled"])
        self.assertFalse(facts["carrier_collected"])

    def test_split_shipments_do_not_supply_an_arbitrary_return_delivery_date(self):
        backend = reader()
        backend.shipments_for_order.return_value.append({"status": "in_transit"})
        facts = BackendFacts(backend, Clock(NOW)).read("return_request", ORDER)
        self.assertNotIn("delivered_at", facts)
        self.assertEqual(facts["evidence_status"], "verified")
        self.assertEqual(facts["reason"], "faulty")

    def test_the_delivery_check_considers_every_parcel_and_says_it_cannot_be_cited(self):
        backend = reader()
        backend.shipments_for_order.return_value.append({"status": "lost"})
        result = self.tools(backend).call("policy", action="check", operation="delivery", order_id=ORDER)
        self.assertTrue(result["requires_human"])
        self.assertEqual(len(result["parcel_checks"]), 2)
        self.assertIn("cannot be cited", result["note"])
        self.assertIn("rule_sources", result)
        self.assertNotIn("policy_sections", result)

    def test_order_level_returns_do_not_establish_the_affected_items(self):
        backend = reader()
        backend.returns_for_order.return_value = {"returns": [{"status": "approved"}], "refunds": []}
        facts = BackendFacts(backend, Clock(NOW)).read("return_request", ORDER, ["ITM-1"])
        self.assertEqual(Rules.load().check("return_request", facts=facts)["check"],
                         "return_request.return_scope_unverified")


class WriteGateTests(unittest.TestCase):
    """The gate on explicit facts, independent of how the records map to them."""

    def setUp(self):
        self.args = {"operation": "refund", "order_id": ORDER, "reason": "damaged_item", "amount": 60,
                     "items": [{"item_id": "ITM-1", "quantity": 1}], "cited_clauses": ["refunds.md"]}

    def gate(self, facts=REFUND, **kwargs):
        tools = services(reader(), writer=FakeDatabase(), **kwargs)
        tools.call("policy", action="read", skill="returns-and-refunds", file="refunds.md")
        patcher = patch.object(tools.gate.facts, "read", return_value=facts)
        patcher.start()
        self.addCleanup(patcher.stop)
        return tools

    def test_auto_review_and_human_routes_come_from_the_rules(self):
        for amount, route in ((20, "auto"), (60, "approver")):
            checked = self.gate().gate.check({**self.args, "amount": amount})
            self.assertEqual((checked["status"], checked["policy_check"]["route"]), ("allowed", route))
        tools = self.gate()
        self.assertEqual(tools.gate.check({**self.args, "amount": 120})["status"], "awaiting_human")
        self.assertTrue(tools.ctx.stopped)

    def test_every_write_needs_a_citation_the_policy_tool_returned(self):
        tools = self.gate()
        for cited, message in (([], "requires at least one cited policy clause"),
                               (["REF-Invented-9.9"], "were not returned by the policy tool"),
                               (["refunds.md", "REF-Invented-9.9"], "were not returned")):
            with self.subTest(cited=cited), self.assertRaisesRegex(ValueError, message):
                tools.gate.check({**self.args, "amount": 20, "cited_clauses": cited})
        other = services(reader(), writer=FakeDatabase())   # another ticket's reads count for nothing
        with patch.object(other.gate.facts, "read", return_value=REFUND), self.assertRaisesRegex(ValueError, "not returned"):
            other.gate.check(self.args)

    def test_the_gate_decides_without_writing_and_the_executor_writes_what_it_is_handed(self):
        tools = self.gate()
        checked = tools.gate.check(self.args)
        self.assertIsNone(tools.gate.review(self.args, checked["policy_check"]))
        self.assertEqual(tools.executor.writer.writes, [])   # deciding never writes
        self.assertIsNone(tools.executor.replay(self.args))
        written = tools.executor.apply(self.args, checked["policy_check"])
        self.assertEqual(written["status"], "executed")
        self.assertNotIn("reason", written["policy_check"])   # the pre-write reason would read as not done
        self.assertTrue(tools.executor.replay(self.args)["replayed"])
        with self.assertRaisesRegex(ValueError, "conflicting amount"):
            tools.executor.replay({**self.args, "amount": 50})

    def test_one_rejection_can_be_fixed_and_the_next_goes_to_a_person(self):
        tools = self.gate(verdicts=["reject", "reject"])
        decision = tools.gate.check(self.args)["policy_check"]
        self.assertEqual(tools.gate.review(self.args, decision)["status"], "blocked")
        self.assertFalse(tools.ctx.stopped)
        self.assertEqual(tools.gate.review(self.args, decision)["status"], "awaiting_human")
        self.assertTrue(tools.ctx.stop_reason.startswith("reviewer reject"))
        self.assertEqual(tools.reviews[0]["write"], self.args)
        self.assertEqual([item["tool"] for item in tools.reviews[0]["evidence"]], ["policy"])

    def test_an_escalation_goes_straight_to_a_person(self):
        tools = self.gate(verdicts=["escalate"])
        decision = tools.gate.check(self.args)["policy_check"]
        self.assertEqual(tools.gate.review(self.args, decision)["status"], "awaiting_human")

    def test_an_approval_covers_only_the_exact_write(self):
        tools = self.gate()
        self.assertEqual(tools.gate.request_review({**self.args, "review_reason": "Per policy."})["status"], "approved")
        self.assertTrue(tools.gate.is_approved(self.args))
        self.assertFalse(tools.gate.is_approved({**self.args, "amount": 59}))
        self.assertFalse(tools.gate.is_approved({**self.args, "items": [{"item_id": "ITM-2", "quantity": 1}]}))
        small = tools.gate.request_review({**self.args, "amount": 20, "review_reason": "Per policy."})
        self.assertEqual(small["status"], "not_required")

    def test_a_review_cannot_override_a_blocked_amount_or_a_human_condition(self):
        for changes, expected in (({"refunded_amount": 190}, "blocked"), ({"disputed": True}, "awaiting_human")):
            with self.subTest(changes=changes):
                tools = self.gate({**REFUND, **changes})
                self.assertEqual(tools.gate.request_review({**self.args, "review_reason": "Override."})["status"],
                                 expected)
                self.assertEqual(tools.reviews, [])

    def test_an_unsupported_operation_goes_to_a_person(self):
        tools = self.gate()
        self.assertEqual(tools.gate.check({**self.args, "operation": "exchange"})["status"], "awaiting_human")
        self.assertEqual(tools.ctx.stop_reason, "unsupported write operation")


if __name__ == "__main__":
    unittest.main()
