"""A reply may not say a pending refund has been paid: checked by code, before the LLM reply check."""
import json
import unittest
from unittest.mock import Mock, patch

from support_ops.backend.postgres import PostgresReader
from support_ops.helpers.config import load_config
from support_ops.helpers.llm import text, tool_call
from support_ops.helpers.reply_rules import internal_id_issue, refund_wording_issue, wording_issues
from support_ops.systems.single_agent import SingleAgentWorkflow
from tests.fake_writer import FakeWriter
from tests.scripted import ScriptedLLM

RECORDED = [{"tool": "backend_write", "result": {"operation": "refund", "status": "executed"}}]


class RefundWordingTests(unittest.TestCase):
    def test_a_pending_refund_described_as_paid_is_caught(self):
        for reply in ("The payment of SGD 32.00 has been refunded to the original Visa card.",
                      "Your refund has been processed.", "The money has been returned to your card.",
                      "Your refund was issued today."):
            with self.subTest(reply=reply):
                self.assertIn("only recorded as pending", refund_wording_issue(reply, RECORDED))

    def test_pending_wording_and_other_tickets_pass(self):
        for reply in ("A refund of SGD 32.00 is recorded and pending.",
                      "Your refund will go back to the original card; the policy estimates 5 to 10 business days."):
            with self.subTest(reply=reply):
                self.assertIsNone(refund_wording_issue(reply, RECORDED))
        self.assertIsNone(refund_wording_issue("Your refund has been processed.", []))   # no refund here
        processed = [{"tool": "get_returns_for_order", "result": {"returns": [], "refunds": [{"status": "processed"}]}}]
        self.assertIsNone(refund_wording_issue("Your refund has been processed.", processed))   # true

    def test_a_pending_refund_that_was_looked_up_counts_too(self):
        pending = [{"tool": "get_returns_for_order", "result": {"returns": [], "refunds": [{"status": "pending"}]}}]
        self.assertIsNotNone(refund_wording_issue("Your refund has been processed.", pending))


class InternalIdTests(unittest.TestCase):
    def test_clause_and_record_ids_are_caught_even_with_non_breaking_dashes(self):
        for reply in ("according to our policy (section 4.2, ODR‑OrderChangeCancel‑4.2), it should appear",
                      "Under REF-Refunds-6.6 you qualify.", "Refund RFD-W-EC7B6EA3E4FA is pending.",
                      "We checked item ITM-EVAL-101-1 and payment PAY-EVAL-109.", "Your ticket is TKT-E139AC2B96F0."):
            with self.subTest(reply=reply):
                self.assertIn("internal IDs", internal_id_issue(reply))

    def test_the_order_number_products_and_plain_policy_wording_pass(self):
        for reply in ("Your order ORD‑2026‑00109 has been cancelled.",
                      "Your Terra Insulated Bottle (SKU AUR-A100) is eligible.",
                      "Under our cancellation policy the refund is pending."):
            with self.subTest(reply=reply):
                self.assertIsNone(internal_id_issue(reply))

    def test_every_broken_rule_is_reported_together(self):
        issues = wording_issues("Your refund has been processed under REF-Refunds-6.6.", RECORDED)
        self.assertEqual(len(issues), 2)


class ReplyLoopTests(unittest.TestCase):
    def test_the_paid_claim_is_redrafted_without_spending_a_reply_check(self):
        cfg = load_config()
        cfg["policy"]["mode"] = "skills"
        facts = {"payment_id": "PAY-1", "payment_status": "captured", "currency": "SGD", "captured_amount": 100,
                 "refunded_amount": 0, "pending_refund_amount": 0, "approved_refund_amount": 100,
                 "request_accepted": True, "inspection_or_waiver": True}
        llm = ScriptedLLM().push("guard", text("0.01")).push(
            "supervisor",
            tool_call("policy", action="read", skill="returns-and-refunds", file="refunds.md"),
            tool_call("backend_write", operation="refund", order_id="ORD-2026-00101", amount=20,
                      reason="damaged_item", cited_clauses=["refunds.md"]),
            text("ready"), text("Your SGD 20 has been refunded to your card."),
            text("A refund of SGD 20 is recorded and pending."),
        )
        workflow = SingleAgentWorkflow(llm, cfg, reader=Mock(spec=PostgresReader), writer=FakeWriter())
        with patch.object(workflow.tools.backend_facts, "read", return_value=facts):
            result = workflow.invoke("T1", "Please refund my bottle.", "react")
        self.assertEqual(result["reply"], "A refund of SGD 20 is recorded and pending.")
        rejected = result["trace"].of_kind("reply_wording_rejected")
        self.assertIn("has been refunded", rejected[0]["issues"][0])
        redraft = json.loads([m for role, m in llm.calls if role == "supervisor"][-1][1]["content"])
        self.assertIn("only recorded as pending", redraft["previous_draft_issues"][0])
        self.assertEqual(len([role for role, _ in llm.calls if role == "approver"]), 1)   # only the good draft


    def test_wording_rewrites_do_not_use_up_the_reply_checks_drafts(self):
        cfg = load_config()
        cfg["policy"]["mode"] = "skills"
        reject = text(json.dumps({"decision": "reject", "issues": ["too vague"], "reason": "x"}))
        llm = ScriptedLLM().push("guard", text("0.01")).push(
            "supervisor", text("ready"), text("Under REF-Refunds-6.6 you qualify."),
            text("Under ODR-OrderChangeCancel-4.2 you qualify."), text("Vague answer."), text("Clear answer."),
        ).push("approver", reject, text(json.dumps({"decision": "approve", "issues": [], "reason": "ok"})))
        result = SingleAgentWorkflow(llm, cfg, reader=Mock(spec=PostgresReader), writer=FakeWriter()).invoke(
            "T1", "Can I return it?", "react")
        self.assertEqual(result["reply"], "Clear answer.")   # 2 wording rewrites, then 2 checked drafts
        self.assertEqual(len(result["trace"].of_kind("reply_wording_rejected")), 2)

if __name__ == "__main__":
    unittest.main()
