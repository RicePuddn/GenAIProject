"""Real writes against an isolated PostgreSQL database, reset around every test.

Set SUPPORT_OPS_TEST_DATABASE_URL to a throwaway database: each test deletes all its rows.
"""
import os
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from support_ops.backend.postgres import PostgresReader, PostgresWriter, WriteRefused
from support_ops.gateway.approvals import TicketFlow
from support_ops.gateway.ticket_state import TicketState
from support_ops.gateway.writes import BackendWrites
from support_ops.helpers.clock import Clock
from support_ops.policy.facts import BackendFacts
from support_ops.policy.routing import RoutingRules
from support_ops.policy.skills import SkillsPolicy
from support_ops.systems.tools import SupportTools
from scripts.seed_backend import seed

DSN = os.environ.get("SUPPORT_OPS_TEST_DATABASE_URL")
NOW = datetime(2026, 10, 31, 9, tzinfo=timezone.utc)
ACCEPTED = "ORD-2026-00101"   # gold-01: an accepted return, SGD 20 refundable
CANCELLABLE = "ORD-2026-00109"   # gold-09: processing, SGD 32 captured


@unittest.skipUnless(DSN, "isolated PostgreSQL URL not configured")
class WriterTests(unittest.TestCase):
    def setUp(self):
        seed(DSN, reset=True)
        self.addCleanup(seed, DSN, reset=True)
        self.writer, self.reader = PostgresWriter(DSN), PostgresReader(DSN)

    def apply(self, operation, order, amount=None, items=(), key="K1", **extra):
        request = {"ticket_id": "T1", "operation": operation, "order_id": order, "amount": amount, "items": list(items)}
        return self.writer.apply(request, idempotency_key=key, reason="damaged", at=NOW, **extra)

    def line(self, order):
        item = self.reader.items_for_order(order)[0]
        return {"item_id": item["id"], "quantity": item["quantity"]}

    def test_a_refund_is_pending_and_counts_against_the_balance(self):
        result = self.apply("refund", ACCEPTED, 20)
        refund = next(row for row in self.reader.returns_for_order(ACCEPTED)["refunds"] if row["id"] == result["write_id"])
        self.assertEqual((refund["status"], refund["amount"], refund["transaction_ref"]), ("pending", Decimal("20.00"), None))
        self.assertEqual(refund["payment_id"], result["payment_id"])

    def test_a_refund_above_the_balance_is_refused_and_writes_nothing(self):
        before = len(self.reader.returns_for_order(ACCEPTED)["refunds"])
        captured = sum(Decimal(str(row["amount"])) for row in self.reader.payments_for_order(ACCEPTED))
        self.apply("refund", ACCEPTED, 20)
        with self.assertRaisesRegex(WriteRefused, "exceeds"):
            self.apply("refund", ACCEPTED, float(captured), key="K2")
        self.assertEqual(len(self.reader.returns_for_order(ACCEPTED)["refunds"]), before + 1)
        self.assertIsNone(self.writer.replay("K2"))

    def test_the_same_key_replays_without_a_second_row(self):
        first = self.apply("refund", ACCEPTED, 20)
        count = len(self.reader.returns_for_order(ACCEPTED)["refunds"])
        again = self.apply("refund", ACCEPTED, 20)
        self.assertTrue(again["replayed"])
        self.assertEqual(again["write_id"], first["write_id"])
        self.assertEqual(len(self.reader.returns_for_order(ACCEPTED)["refunds"]), count)
        self.assertEqual(self.writer.replay("K1")["result"]["write_id"], first["write_id"])

    def test_record_ids_follow_the_key_so_reruns_match(self):
        first = self.apply("refund", ACCEPTED, 20)["write_id"]
        seed(DSN, reset=True)
        self.assertEqual(self.apply("refund", ACCEPTED, 20)["write_id"], first)

    def test_a_return_writes_the_request_and_its_lines(self):
        line = self.line(CANCELLABLE)
        result = self.apply("return", CANCELLABLE, items=[line])
        request = next(row for row in self.reader.returns_for_order(CANCELLABLE)["returns"] if row["id"] == result["write_id"])
        self.assertEqual((request["status"], request["evidence_status"], request["inspection_status"]),
                         ("requested", "pending", "pending"))
        self.assertEqual([(row["order_item_id"], row["quantity"]) for row in request["items"]],
                         [(line["item_id"], line["quantity"])])

    def test_lines_must_be_on_the_order_in_the_quantity_ordered(self):
        line = self.line(CANCELLABLE)
        for bad in ({"item_id": "ITM-NOPE", "quantity": 1}, {**line, "quantity": line["quantity"] + 1}):
            with self.subTest(bad), self.assertRaisesRegex(WriteRefused, "is not on order"):
                self.apply("return", CANCELLABLE, items=[bad])
        self.assertEqual(self.reader.returns_for_order(CANCELLABLE)["returns"], [])

    def test_an_exchange_needs_valid_lines_and_is_recorded_only_as_a_write(self):
        result = self.apply("exchange", CANCELLABLE, items=[self.line(CANCELLABLE)])
        self.assertEqual(self.reader.returns_for_order(CANCELLABLE)["returns"], [])
        self.assertEqual(self.writer.replay("K1")["result"]["write_id"], result["write_id"])

    def test_cancelling_updates_the_order_lines_and_shipments_but_refunds_nothing(self):
        self.apply("order_change", CANCELLABLE)
        self.assertEqual(self.reader.read("orders", CANCELLABLE)["status"], "cancelled")
        self.assertEqual({row["item_status"] for row in self.reader.items_for_order(CANCELLABLE)}, {"cancelled"})
        self.assertEqual({row["status"] for row in self.reader.shipments_for_order(CANCELLABLE)}, {"cancelled"})
        self.assertEqual(self.reader.returns_for_order(CANCELLABLE)["refunds"], [])

    def test_only_a_processing_order_can_be_cancelled(self):
        self.apply("order_change", CANCELLABLE)
        with self.assertRaisesRegex(WriteRefused, "cannot be cancelled"):
            self.apply("order_change", CANCELLABLE, key="K2")

    def test_a_cancelled_order_is_owed_its_captured_balance_under_the_rules(self):
        rules = RoutingRules.load()
        facts = BackendFacts(self.reader, Clock(NOW))
        self.apply("order_change", CANCELLABLE)
        minimum = rules.min_route("refund", 32, business_facts=facts.read("refund", CANCELLABLE))
        self.assertEqual((minimum["allowed"], minimum["route"]), (True, "approver"))
        self.assertFalse(rules.min_route("refund", 33, business_facts=facts.read("refund", CANCELLABLE))["allowed"])
        self.apply("refund", CANCELLABLE, 32, key="K2")
        after = rules.min_route("refund", 1, business_facts=facts.read("refund", CANCELLABLE))
        self.assertFalse(after["allowed"])   # the pending refund used up the balance

    def test_a_refund_on_an_order_that_is_not_cancelled_gets_no_cancellation_path(self):
        facts = BackendFacts(self.reader, Clock(NOW)).read("refund", CANCELLABLE)
        self.assertIsNot(facts.get("request_accepted"), True)

    def test_the_single_agent_tool_cancels_and_refunds_as_two_real_writes(self):
        clock = Clock(NOW)
        tools = SupportTools(self.reader, SkillsPolicy(), TicketState(clock), self.writer,
                             reviewer=lambda request: {"decision": "approve", "issues": []})
        for file in ("SKILL.md", "refunds.md"):
            tools.execute("T1", "policy", {"action": "read", "skill": "returns-and-refunds", "file": file})
        cancel = tools.execute("T1", "backend_write", {"operation": "cancel", "order_id": CANCELLABLE,
                                                       "reason": "order_cancellation",
                                                       "cited_clauses": ["returns-and-refunds/SKILL.md"]})
        self.assertEqual((cancel["status"], self.reader.read("orders", CANCELLABLE)["status"]), ("executed", "cancelled"))
        args = {"operation": "refund", "order_id": CANCELLABLE, "amount": 32, "reason": "order_cancellation",
                "cited_clauses": ["refunds.md"]}
        self.assertEqual(tools.execute("T1", "backend_write", args)["status"], "review_required")
        refund = tools.execute("T1", "backend_write", {**args, "review_reason": "Cancelled before fulfilment; "
                                                       "the captured balance is owed under 4.2."})
        self.assertEqual(refund["status"], "executed")
        self.assertEqual([(row["status"], row["amount"]) for row in self.reader.returns_for_order(CANCELLABLE)["refunds"]],
                         [("pending", Decimal("32.00"))])
        blocked = tools.execute("T2", "backend_write", {**args, "review_reason": "Again."})
        self.assertEqual(blocked["status"], "blocked")

    def ticket(self, **overrides):
        fields = dict(ticket_id="TKT-NEW", status="in_progress", category="refund", priority="normal",
                      subject="Refund asked.", customer_message="Refund please", at=NOW)
        return self.writer.save_ticket(**{**fields, **overrides})

    def test_a_ticket_with_an_unknown_order_links_no_order(self):
        row = self.ticket(order_id="ORD-NOPE")
        self.assertEqual((row["customer_id"], row["order_id"], row["channel"]), ("dummy_00", None, "chat"))
        self.assertEqual(self.reader.read("support_tickets", "TKT-NEW")["assigned_team"], "returns_team")

    def test_a_ticket_is_updated_in_place_and_keeps_the_placeholder_customer_even_with_an_order(self):
        self.ticket()
        later = datetime(2026, 10, 31, 10, tzinfo=timezone.utc)
        row = self.ticket(order_id=ACCEPTED, status="awaiting_human", customer_message="Second message", at=later,
                          handoff_packet={"reason": "route: human"}, assigned_team="payments_team")
        self.assertEqual((row["customer_id"], row["order_id"]), ("dummy_00", ACCEPTED))
        self.assertEqual((row["status"], row["handoff_packet"]["reason"], row["assigned_team"]),
                         ("awaiting_human", "route: human", "payments_team"))
        self.assertEqual((row["customer_message"], row["created_at"], row["updated_at"]), ("Refund please", NOW, later))
        resolved = self.ticket(status="resolved", resolution="Refund recorded.")
        self.assertEqual((resolved["order_id"], resolved["resolution"]), (row["order_id"], "Refund recorded."))
        self.assertEqual(resolved["handoff_packet"], {"reason": "route: human"})

    def test_a_signed_in_ticket_is_the_customers_and_another_customer_cannot_take_it_over(self):
        row = self.ticket(order_id=ACCEPTED, customer_id="CUS-EVAL-001")
        self.assertEqual((row["customer_id"], row["order_id"]), ("CUS-EVAL-001", ACCEPTED))
        with self.assertRaisesRegex(WriteRefused, "belongs to another customer"):
            self.ticket(status="resolved", customer_id="CUS-1001")
        self.assertEqual(self.reader.read("support_tickets", "TKT-NEW")["status"], "in_progress")
        other = self.ticket(ticket_id="TKT-OTHER", order_id=ACCEPTED, customer_id="CUS-1001")
        self.assertIsNone(other["order_id"])   # another customer's order is never linked

    def test_a_signed_in_write_to_another_customers_order_is_refused_as_not_found(self):
        with self.assertRaisesRegex(WriteRefused, f"order {CANCELLABLE} not found"):
            self.apply("order_change", CANCELLABLE, customer_id="CUS-1001")
        self.assertEqual(self.reader.read("orders", CANCELLABLE)["status"], "processing")
        self.assertEqual(self.apply("order_change", CANCELLABLE, customer_id="CUS-EVAL-001")["status"], "executed")

    def test_the_customer_list_leaves_out_the_placeholder(self):
        ids = [row["id"] for row in self.reader.customers()]
        self.assertIn("CUS-EVAL-001", ids)
        self.assertNotIn("dummy_00", ids)

    def test_the_update_ticket_tool_saves_the_conversation(self):
        tools = SupportTools(self.reader, SkillsPolicy(), TicketState(Clock(NOW)), self.writer)
        result = tools.execute("TKT-S1", "update_ticket", {
            "status": "resolved", "category": "cancellation", "subject": "Cancelled by mistake.",
            "customer_message": "Please cancel", "order_id": CANCELLABLE, "resolution": "Cancelled."})
        self.assertEqual((result["status"], result["ticket_status"], result["order_id"]),
                         ("ticket_saved", "resolved", CANCELLABLE))
        self.assertEqual(self.reader.read("support_tickets", "TKT-S1")["assigned_team"], "order_support")

    def test_a_refused_database_write_comes_back_as_refused_through_the_gateway(self):
        clock = Clock(NOW)
        flow = TicketFlow(TicketState(clock), clock)
        writes = BackendWrites(flow, self.writer)
        flow.statuses["T1"] = "approved"
        flow.tokens["tok"] = {"ticket_id": "T1", "operation": "order_change", "order_id": ACCEPTED, "amount": None,
                              "items": [], "route": "auto", "payment_id": None, "return_id": None,
                              "expires_at": NOW, "used": False}
        result = writes.execute("T1", token="tok", operation="order_change", order_id=ACCEPTED, idempotency_key="K1")
        self.assertEqual(result["status"], "refused")
        self.assertIn("cannot be cancelled", result["reason"])
        self.assertEqual(flow.status("T1"), "approved")


if __name__ == "__main__":
    unittest.main()
