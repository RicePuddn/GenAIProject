"""Shared hard rules and the pluggable policy provider that separates S1 (RAG) from S2 (skills)."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Literal
from unittest.mock import Mock, patch

from support_ops.backend.postgres import PostgresReader
from support_ops.helpers.arguments import NonEmpty, ToolArguments
from support_ops.helpers.config import load_config
from support_ops.helpers.llm import ScriptedLLM, text, tool_call
from support_ops.policy.rules import Rules
from support_ops.policy.rag.loader import load_corpus
from support_ops.helpers.config import ROOT
from support_ops.systems.single_agent import SingleAgentWorkflow

REFUND_FACTS = {
    "payment_status": "captured", "currency": "SGD", "captured_amount": "150.00",
    "refunded_amount": "0.00", "pending_refund_amount": "0.00", "approved_refund_amount": "100.00",
    "original_payment_method": True, "request_accepted": True, "inspection_or_waiver": True,
}


def write_rules(directory: str, limit: str) -> Path:
    data = json.loads((ROOT / "src/support_ops/policy/rules/rules.json").read_text())
    data["groups"]["refund"].update(rule_id="test_limit", max_reviewed_amount=limit, max_auto_amount="10.00")
    path = Path(directory) / "rules.json"
    path.write_text(json.dumps(data))
    return path


class RulesTests(unittest.TestCase):
    def test_refund_limit_boundary_comes_from_rules_file(self):
        rules = Rules.load()
        self.assertEqual(rules.refund_limit, 100)
        decision = rules.check("refund", 100, REFUND_FACTS)
        self.assertTrue(decision["allowed"])
        self.assertEqual((decision["rule"], decision["limit"], decision["currency"]),
                         ("refund_approval_requirements", 100.0, "SGD"))
        above = rules.check("refund", 100.01)
        self.assertFalse(above["allowed"])
        self.assertTrue(above["requires_human"])

    def test_unknown_operations_missing_facts_and_invalid_amounts(self):
        rules = Rules.load()
        self.assertTrue(rules.check("cancel")["requires_human"])
        self.assertTrue(rules.check("exchange")["requires_human"])
        for amount in (None, 0, -1, float("nan"), float("inf")):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                rules.check("refund", amount)

    def test_invalid_rules_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for limit in ("0", "-5", "abc"):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    Rules.load(write_rules(directory, limit))
            path = Path(directory) / "missing.json"
            path.write_text(json.dumps({"version":"x"}))
            with self.assertRaises(ValueError):
                Rules.load(path)

    def test_refund_requires_verified_financial_and_approval_facts(self):
        rules = Rules.load()
        self.assertTrue(rules.check("refund", 50)["requires_human"])
        for change in ({"payment_status": "failed"}, {"original_payment_method": False},
                       {"request_accepted": False}, {"inspection_or_waiver": False},
                       {"pending_refund_amount": 110}, {"approved_refund_amount": 40}):
            with self.subTest(change=change):
                result = rules.check("refund", 50, {**REFUND_FACTS, **change})
                self.assertFalse(result["allowed"])
                self.assertEqual(result["requires_human"], not any(
                    key in change for key in ("pending_refund_amount", "approved_refund_amount")))

    def test_return_window_day_30_is_included_and_late_requests_get_review(self):
        rules = Rules.load()
        facts = {"delivered_at": "2026-10-01", "requested_at": "2026-10-31",
                 "reason": "faulty", "evidence_status": "verified"}
        self.assertTrue(rules.check("return_request", facts=facts)["allowed"])
        self.assertTrue(rules.check("return_request", facts={**facts, "requested_at": "2026-11-01"})["requires_human"])
        # UTC delivery is 2 October in Singapore, so 1 November is still day 30.
        result = rules.check("return_request", facts={**facts, "delivered_at": "2026-10-01T16:30:00Z",
                                                     "requested_at": "2026-11-01"})
        self.assertTrue(result["allowed"])
        self.assertEqual(result["deadline"], "2026-11-01")

    def test_final_sale_and_evidence_exceptions_do_not_become_automatic_rejections(self):
        rules = Rules.load()
        facts = {"delivered_at": "2026-10-01", "requested_at": "2026-10-02", "reason": "change_of_mind",
                 "evidence_status": "verified", "final_sale": True, "unused": True,
                 "original_packaging": True, "tags_complete": True, "accessories_complete": True}
        rejected = rules.check("return_request", facts=facts)
        self.assertFalse(rejected["allowed"])
        self.assertFalse(rejected["requires_human"])
        self.assertTrue(rules.check("return_request", facts={**facts, "reason": "faulty", "original_packaging": False})["allowed"])
        for change in ({"evidence_status": "unavailable"}, {"evidence_status": "inaccessible"},
                       {"evidence_status": "unsafe"}, {"reason": "counterfeit"}, {"disputed": True}):
            with self.subTest(change=change):
                result = rules.check("return_request", facts={**facts, **change})
                self.assertFalse(result["allowed"])
                self.assertTrue(result["requires_human"])

    def test_return_dispatch_requires_handover_and_an_approved_extension(self):
        rules = Rules.load()
        facts = {"approved_at": "2026-10-01", "handed_over_at": "2026-10-08", "carrier_accepted": True}
        self.assertTrue(rules.check("return_dispatch", facts=facts)["allowed"])
        late = {**facts, "handed_over_at": "2026-10-09"}
        self.assertTrue(rules.check("return_dispatch", facts=late)["requires_human"])
        extended = {**late, "extension_until": "2026-10-10", "extension_approved": True}
        self.assertTrue(rules.check("return_dispatch", facts=extended)["allowed"])
        self.assertTrue(rules.check("return_dispatch", facts={**extended, "extension_approved": False})["requires_human"])
        self.assertTrue(rules.check("return_dispatch", facts={**facts, "carrier_accepted": False})["requires_human"])

    def test_cancellation_checks_label_packing_fulfillment_and_pickup(self):
        rules = Rules.load()
        facts = {"order_status": "processing", "authority_verified": True,
                 **{name: False for name in rules.data["cancel"]["blockers"]}}
        self.assertTrue(rules.check("cancel", facts=facts)["allowed"])
        for name in rules.data["cancel"]["blockers"]:
            with self.subTest(blocker=name):
                self.assertTrue(rules.check("cancel", facts={**facts, name: True})["requires_human"])
        self.assertTrue(rules.check("cancel", facts={"order_status": "processing"})["requires_human"])

    def test_delivery_escalation_is_after_day_3_with_immediate_exceptions(self):
        rules = Rules.load()
        facts = {"delivery_status": "in_transit", "estimated_delivery": "2026-10-01", "as_of": "2026-10-04"}
        self.assertFalse(rules.check("delivery", facts=facts)["requires_human"])
        self.assertTrue(rules.check("delivery", facts={**facts, "as_of": "2026-10-05"})["requires_human"])
        for status in rules.data["delivery"]["immediate_human_statuses"]:
            with self.subTest(status=status):
                self.assertTrue(rules.check("delivery", facts={"delivery_status": status})["requires_human"])

    def test_rules_reference_existing_clauses_and_match_the_policy_numbers(self):
        rules = Rules.load()
        clauses = {c.id: c.text for c in load_corpus(ROOT / "policy/raw_policy/pdf")}
        for key in ("refund", "return_request", "return_dispatch", "cancel", "delivery", "human_review"):
            self.assertTrue(set(rules.data[key]["sources"]) <= clauses.keys())
        for key, section, field in (("return_request", "RET-ReturnEligibility-2.1", "window_calendar_days"),
                                    ("return_dispatch", "RET-ReturnProcedure-4.2", "window_calendar_days"),
                                    ("delivery", "SHP-DeliveryProblems-7.1", "investigate_after_calendar_days")):
            days = int(re.search(r"\((\d+)\) calendar days", clauses[section]).group(1))
            self.assertEqual(days, rules.data[key][field])
        amount = max(float(value) for value in re.findall(r"SGD ([0-9.]+)", clauses["REF-Refunds-6.6"]))
        self.assertEqual(float(amount), float(rules.refund_limit))

    def test_policy_change_blocks_stale_rules_until_they_are_reviewed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = json.loads((ROOT / "src/support_ops/policy/rules/rules.json").read_text())
            folder = root / "policy/raw_policy/pdf"
            folder.mkdir(parents=True)
            for name in data["policy_documents"]:
                (folder / name).write_bytes((ROOT / "policy/raw_policy/pdf" / name).read_bytes())
            (folder / "returns-and-refunds-policy.pdf").write_bytes(b"changed")
            path = root / "rules.json"
            path.write_text(json.dumps(data))
            with patch("support_ops.policy.rules.rules.ROOT", root), self.assertRaisesRegex(ValueError, "review its hard rules"):
                Rules.load(path)


    def test_json_outcomes_are_used_and_both_true_is_rejected(self):
        original = json.loads((ROOT / "src/support_ops/policy/rules/rules.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            outcome = original["groups"]["cancel"]["checks"]["already_cancelled"]["outcome"]
            outcome.update(requires_human=True, route="human", reason="Review the recorded cancellation.")
            path.write_text(json.dumps(original))
            result = Rules.load(path).check("cancel", facts={"order_status":"cancelled"})
            self.assertEqual(result["check"], "cancel.already_cancelled")
            self.assertTrue(result["requires_human"])
            self.assertEqual(result["reason"], outcome["reason"])
            outcome["allowed"] = True
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "together"):
                Rules.load(path)

    def test_currency_review_is_separate_from_numeric_input_validation(self):
        rules = Rules.load()
        result = rules.check("refund", 50, {**REFUND_FACTS, "currency":"USD"})
        self.assertEqual(result["check"], "refund.currency_mismatch")
        self.assertFalse(result["allowed"])
        self.assertTrue(result["requires_human"])
        self.assertTrue(rules.check("refund", "50.00", REFUND_FACTS)["allowed"])
        with self.assertRaises(ValueError):
            rules.check("refund", "not-a-number", REFUND_FACTS)
        with self.assertRaises(ValueError):
            rules.check("return_request", facts={"delivered_at":"invalid", "requested_at":"2026-10-31"})


class FakeArguments(ToolArguments):
    """search: find policy clauses for a query."""
    action: Literal["search"]
    query: NonEmpty


class FakePolicy:
    arguments = FakeArguments
    description = FakeArguments.__doc__
    prompt_hint = "Search the policy before advising."

    def __init__(self):
        self.calls = []

    def planner_context(self) -> str:
        return ""

    def run(self, args):
        self.calls.append(args)
        return {"clauses": [{"id": "RET-1", "text": "30-day returns."}]}


class WorkflowWiringTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        build = patch("support_ops.policy.rag.policy.shared_retriever", return_value=Mock())
        build.start()
        self.addCleanup(build.stop)
        self.reader = Mock(spec=PostgresReader)

    def test_configured_rules_reach_both_the_gateway_and_the_policy_check(self):
        llm = ScriptedLLM().push("guard", text("0.01")).push(
            "supervisor",
            tool_call("policy", action="check", operation="refund", amount=30),
            tool_call("ticket_state", action="write", operation="refund", record_id="O1", amount=30),
        )
        with tempfile.TemporaryDirectory() as directory:
            self.cfg["policy"]["rules_file"] = str(write_rules(directory, "20"))
            result = SingleAgentWorkflow(llm, self.cfg, reader=self.reader).invoke("T1", "Refund 30?", "react")
        check = result["evidence"][0]["result"]
        self.assertEqual((check["rule"], check["limit"], check["allowed"]), ("test_limit", 20.0, False))
        self.assertEqual(result["ticket_status"], "awaiting_human")

    def test_unknown_policy_mode_is_rejected(self):
        self.cfg["policy"]["mode"] = "telepathy"
        with self.assertRaises(ValueError):
            SingleAgentWorkflow(ScriptedLLM(), self.cfg, reader=self.reader)

    def test_a_different_policy_provider_changes_only_the_policy_tool_and_hint(self):
        policy = FakePolicy()
        llm = ScriptedLLM().push("guard", text("0.01")).push(
            "supervisor", tool_call("policy", action="search", query="return window"),
            text("ready"), text("You have 30 days."),
        )
        result = SingleAgentWorkflow(llm, self.cfg, reader=self.reader, policy=policy).invoke(
            "T1", "Can I return it?", "react")
        self.assertEqual(policy.calls, [{"action": "search", "query": "return window"}])
        self.assertEqual(result["evidence"][0]["result"]["clauses"][0]["id"], "RET-1")
        system = llm.calls[1][1][0]["content"]
        self.assertIn("Search the policy before advising.", system)
        self.assertNotIn("SKILL.md", system)

        planner = ScriptedLLM().push("guard", text("0.01")).push(
            "supervisor", text(json.dumps({"calls": []})), text("Hello."))
        SingleAgentWorkflow(planner, self.cfg, reader=self.reader, policy=policy).invoke(
            "T2", "Hi", "plan_and_execute")
        prompt = planner.calls[1][1][0]["content"]
        self.assertIn("search: find policy clauses", prompt)
        self.assertNotIn("Available policy filenames", prompt)
        self.assertIn('"title": "FakeArguments"', prompt)
        self.assertNotIn('"title": "PolicyArguments"', prompt)


if __name__ == "__main__":
    unittest.main()
