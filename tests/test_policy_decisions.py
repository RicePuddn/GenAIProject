"""Existing backend fields feed rules; unrecorded facts remain unknown."""
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from support_ops.gateway.ticket_state import TicketState
from support_ops.helpers.clock import Clock
from support_ops.policy.facts import BackendFacts
from support_ops.policy.rag import RagPolicy
from support_ops.policy.skills import SkillsPolicy
from support_ops.systems.tools import SupportTools
from tests.fake_writer import FakeWriter
from tests.test_multi_agent_gateway import NOW, ORDER, reader
from tests.test_routing import REFUND


class BackendFactTests(unittest.TestCase):
    def tools(self, provider="skills", backend=None, reviewer=None):
        policy = SkillsPolicy() if provider == "skills" else RagPolicy(Mock())
        return SupportTools(backend or reader(), policy, TicketState(Clock(NOW)), FakeWriter(), reviewer=reviewer)

    @staticmethod
    def retrieve(tools, provider="skills"):
        """Retrieve the refunds policy in ticket T1 and return a citation for it."""
        if provider == "skills":
            tools.execute("T1", "policy", {"action": "read", "skill": "returns-and-refunds", "file": "refunds.md"})
            return "refunds.md"
        tools.policy.retriever.search.return_value = [{"id": "REF-Refunds-6.6", "text": "Refund rules."}]
        tools.execute("T1", "policy", {"action": "search", "query": "refund"})
        return "REF-Refunds-6.6"

    def write(self, tools, amount=20, **details):
        """A refund in ticket T1; unless the test names its citations, it retrieves and cites the refunds policy."""
        if "cited_clauses" not in details:
            details["cited_clauses"] = [self.retrieve(tools, "skills" if isinstance(tools.policy, SkillsPolicy) else "rag")]
        return tools.execute("T1", "backend_write", {
            "operation": "refund", "order_id": ORDER, "reason": "damaged_item",
            "items": [{"item_id": "ITM-1", "quantity": 1}], "amount": amount, **details,
        })

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

    def test_both_providers_require_human_review_for_unrecorded_prerequisites(self):
        for provider in ("rag", "skills"):
            with self.subTest(provider=provider):
                backend = reader()
                backend.returns_for_order.return_value = {"returns": [], "refunds": []}
                tools = self.tools(provider, backend)
                result = tools.execute("T1", "policy", {
                    "action": "check", "operation": "refund", "order_id": ORDER, "amount": 20,
                })
                self.assertEqual(result["check"], "refund.missing_facts")
                self.assertIn("inspection_or_waiver", result["missing_facts"])
                self.assertEqual(self.write(tools, review_reason="I say it passed.")["status"], "awaiting_human")
                self.assertFalse(tools.writer.applied)
                with self.assertRaises(ValidationError):
                    self.write(tools, facts={"inspection_or_waiver": True})

    def test_unknown_item_ids_cannot_use_another_items_records(self):
        tools = self.tools()
        result = self.write(tools, items=[{"item_id": "OTHER-ITEM", "quantity": 1}])
        self.assertEqual(result["status"], "awaiting_human")

    def test_refund_cannot_reuse_an_approval_already_paid_or_pending(self):
        for status in ("processed", "pending"):
            for provider in ("rag", "skills"):
                with self.subTest(status=status, provider=provider):
                    backend = reader(total=100, refunds=[{
                        "return_id": "RET-APPROVED", "status": status, "amount": "20.00",
                    }])
                    request = backend.returns_for_order.return_value["returns"][0]
                    request["items"] = [{**request["items"][0], "approved_refund_amount": "20.00"}]
                    tools = self.tools(provider, backend)
                    result = self.write(tools)
                    self.assertEqual(result["status"], "blocked")
                    self.assertEqual(result["policy_check"]["check"], "refund.approved_amount_exceeded")
                    self.assertFalse(tools.writer.applied)

    def test_remaining_approval_excludes_failed_and_other_return_refunds(self):
        backend = reader(total=100, refunds=[
            {"return_id": "RET-APPROVED", "status": "processed", "amount": "10.00"},
            {"return_id": "RET-APPROVED", "status": "failed", "amount": "20.00"},
            {"return_id": "RET-OTHER", "status": "processed", "amount": "5.00"},
        ])
        request = backend.returns_for_order.return_value["returns"][0]
        request["items"] = [{**request["items"][0], "approved_refund_amount": "30.00"}]
        tools = self.tools(backend=backend)
        facts = tools.backend_facts.read("refund", ORDER, ["ITM-1"])
        self.assertEqual(facts["approved_refund_amount"], 20)
        self.assertEqual(self.write(tools)["status"], "executed")

    def test_partial_refund_allocation_stays_unknown_for_a_multi_item_return(self):
        backend = reader(total=100, refunds=[{
            "return_id": "RET-APPROVED", "status": "processed", "amount": "20.00",
        }])
        tools = self.tools(backend=backend)
        result = self.write(tools)
        self.assertEqual(result["status"], "awaiting_human")
        self.assertFalse(tools.writer.applied)

    def test_open_item_states_and_no_shipments_allow_pre_fulfillment_cancellation(self):
        backend = reader()
        backend.shipments_for_order.return_value = []
        for item in backend.items_for_order.return_value:
            item["item_status"] = "open"
        facts = BackendFacts(backend, Clock(NOW)).read("cancel", ORDER)
        self.assertEqual(facts, {"order_status": "processing", "any_quantity_fulfilled": False,
                                 "packing_started": False, "label_created": False, "carrier_collected": False})
        result = self.tools(backend=backend).execute("T1", "policy", {
            "action": "check", "operation": "cancel", "order_id": ORDER,
        })
        self.assertTrue(result["allowed"])

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

    def test_delivery_check_considers_every_parcel(self):
        backend = reader()
        backend.shipments_for_order.return_value.append({"status": "lost"})
        result = self.tools(backend=backend).execute("T1", "policy", {
            "action": "check", "operation": "delivery", "order_id": ORDER,
        })
        self.assertTrue(result["requires_human"])
        self.assertEqual(len(result["parcel_checks"]), 2)

    def test_order_level_returns_do_not_establish_the_affected_items(self):
        backend = reader()
        backend.returns_for_order.return_value = {"returns": [{"status": "approved"}], "refunds": []}
        result = self.tools(backend=backend).execute("T1", "policy", {
            "action": "check", "operation": "return_request", "order_id": ORDER, "item_ids": ["ITM-1"],
        })
        self.assertEqual(result["check"], "return_request.return_scope_unverified")

    # Supply complete inputs explicitly to test the write gate independently of backend mapping.
    def test_write_gate_still_enforces_auto_review_and_human_routes(self):
        for provider in ("rag", "skills"):
            with self.subTest(provider=provider):
                tools = self.tools(provider)
                with patch.object(tools.backend_facts, "read", return_value=REFUND):
                    self.assertEqual(self.write(tools)["policy_check"]["route"], "auto")
                requests = []
                approve = lambda request: requests.append(request) or {"decision": "approve", "issues": []}
                tools = self.tools(provider, reviewer=approve)
                cited = self.retrieve(tools, provider)
                with patch.object(tools.backend_facts, "read", return_value=REFUND):
                    self.assertEqual(self.write(tools, 60)["status"], "review_required")
                    for amount in (20, 60):   # auto and approver routes alike need a citation
                        with self.assertRaisesRegex(ValueError, "requires at least one cited policy clause"):
                            self.write(tools, amount, review_reason="Reviewed.", cited_clauses=[])
                    self.assertFalse(tools.writer.applied)
                    reviewed = self.write(tools, 60, review_reason="Reviewed the policy, facts and amount.",
                                          cited_clauses=[cited])
                    self.assertEqual(reviewed["policy_check"]["route"], "approver")
                    self.assertEqual(reviewed["status"], "executed")
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0]["proposal"]["cited_clauses"], [cited])
                self.assertEqual(requests[0]["evidence"][0]["tool"], "policy")
                tools = self.tools(provider)
                with patch.object(tools.backend_facts, "read", return_value=REFUND):
                    self.assertEqual(self.write(tools, 120, review_reason="Please allow.")["status"], "awaiting_human")

    def test_a_write_may_cite_only_what_the_policy_tool_returned_in_this_ticket(self):
        for provider in ("rag", "skills"):
            with self.subTest(provider=provider):
                tools = self.tools(provider, reviewer=lambda request: {"decision": "approve", "issues": []})
                with patch.object(tools.backend_facts, "read", return_value=REFUND):
                    with self.assertRaisesRegex(ValueError, "were not returned by the policy tool"):
                        self.write(tools, 60, review_reason="Per policy.", cited_clauses=["REF-Refunds-6.6"])
                    cited = self.retrieve(tools, provider)
                    with self.assertRaisesRegex(ValueError, "were not returned"):   # another ticket's reads
                        tools.execute("T2", "backend_write", {
                            "operation": "refund", "order_id": ORDER, "reason": "damaged_item", "amount": 60,
                            "items": [{"item_id": "ITM-1", "quantity": 1}], "review_reason": "Per policy.",
                            "cited_clauses": [cited]})
                    with self.assertRaisesRegex(ValueError, "were not returned"):   # checked on auto writes too
                        self.write(tools, cited_clauses=[cited, "REF-Invented-9.9"])
                self.assertFalse(tools.writer.applied)

    def test_the_gate_decides_without_writing_and_the_executor_writes_what_it_is_handed(self):
        tools = self.tools(reviewer=lambda request: {"decision": "approve", "issues": []})
        args = {"operation": "refund", "order_id": ORDER, "reason": "damaged_item", "amount": 60,
                "items": [{"item_id": "ITM-1", "quantity": 1}], "review_reason": "Per policy.",
                "cited_clauses": [self.retrieve(tools)]}
        with patch.object(tools.backend_facts, "read", return_value=REFUND):
            checked = tools.gate.check("T1", args)
        self.assertEqual((checked["status"], checked["policy_check"]["route"]), ("allowed", "approver"))
        self.assertIsNone(tools.gate.review("T1", args, checked["policy_check"]))
        self.assertFalse(tools.writer.applied)   # deciding never writes
        self.assertIsNone(tools.executor.replay("T1", args, checked["policy_check"]))
        self.assertEqual(tools.executor.apply("T1", args, checked["policy_check"])["status"], "executed")
        self.assertTrue(tools.executor.replay("T1", args, checked["policy_check"])["replayed"])

    def test_the_independent_review_decides_writes_requiring_review(self):
        verdicts = {"reject": {"decision": "reject", "issues": ["amount exceeds the item paid"]},
                    "escalate": {"decision": "escalate", "issues": ["needs an operator"]}}
        for name, verdict in verdicts.items():
            with self.subTest(verdict=name):
                tools = self.tools(reviewer=lambda request: verdict)
                cited = self.retrieve(tools)
                with patch.object(tools.backend_facts, "read", return_value=REFUND):
                    first = self.write(tools, 60, review_reason="Per policy.", cited_clauses=[cited])
                    if name == "reject":   # the agent may fix one rejection; the next goes to a human
                        self.assertEqual(first["status"], "blocked")
                        self.assertIn("amount exceeds the item paid", first["reason"])
                        first = self.write(tools, 60, review_reason="Per policy.", cited_clauses=[cited])
                    self.assertEqual(first["status"], "awaiting_human")
                    self.assertIn(f"reviewer {name}", first["handoff_packet"]["reason"])
                self.assertFalse(tools.writer.applied)
        tools = self.tools()   # no reviewer: a write requiring review is never executed unreviewed
        cited = self.retrieve(tools)
        with patch.object(tools.backend_facts, "read", return_value=REFUND):
            self.assertEqual(self.write(tools, 60, review_reason="Per policy.", cited_clauses=[cited])["status"],
                             "awaiting_human")
        self.assertFalse(tools.writer.applied)

    def test_review_cannot_override_a_blocked_amount_or_human_condition(self):
        for changes, expected in (({"refunded_amount": 190}, "blocked"), ({"disputed": True}, "awaiting_human")):
            with self.subTest(changes=changes):
                tools = self.tools()
                with patch.object(tools.backend_facts, "read", return_value={**REFUND, **changes}):
                    self.assertEqual(self.write(tools, review_reason="Override.")["status"], expected)
                self.assertFalse(tools.writer.applied)
