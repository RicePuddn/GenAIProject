"""review.trigger: who starts the independent review of a write requiring one. The rules are the same
either way; only the trigger differs. The pure supervisor's specialists (S3/S4) use agent; the single
agent keeps the gate's default, automatic."""
import unittest

from support_ops.gateway.ticket_state import TicketState
from support_ops.helpers.clock import Clock
from support_ops.helpers.config import load_config
from support_ops.policy.rules import Rules
from support_ops.policy.skills import SkillsPolicy
from support_ops.systems.tools import SupportTools
from tests.fake_writer import FakeWriter
from tests.test_multi_agent import NOW, ORDER, reader
from tests.test_multi_agent_skills import skills_dir

REFUND = {"operation": "refund", "order_id": ORDER, "items": [{"item_id": "ITM-0031-2", "quantity": 2}],
          "reason": "damaged_item", "cited_clauses": ["returns-and-refunds/refunds.md"]}


class WriteGateTriggerTests(unittest.TestCase):
    def tools(self, trigger, *verdicts):
        """Shared tools with a scripted reviewer, after reading the policy the writes cite."""
        self.reviews = []
        answers = iter(verdicts or [{"decision": "approve"}] * 5)
        tools = SupportTools(reader(), SkillsPolicy(skills_dir()), TicketState(Clock(NOW), rules=Rules.load()),
                             FakeWriter(), reviewer=lambda request: self.reviews.append(request) or next(answers),
                             review_trigger=trigger)
        tools.execute("T1", "policy", {"action": "read", "skill": "returns-and-refunds", "file": "refunds.md"})
        return tools

    def test_agent_trigger_refuses_the_write_until_that_exact_write_is_approved(self):
        tools = self.tools("agent")
        self.assertEqual(tools.execute("T1", "backend_write", {**REFUND, "amount": 50})["status"], "review_required")
        self.assertEqual(self.reviews, [])   # the write never starts the review itself
        approved = tools.execute("T1", "request_review", {**REFUND, "amount": 60, "review_reason": "Damaged."})
        self.assertEqual(approved["status"], "approved")
        # An approval covers only its own amount and items.
        self.assertEqual(tools.execute("T1", "backend_write", {**REFUND, "amount": 50})["status"], "review_required")
        tools.execute("T1", "request_review", {**REFUND, "amount": 50, "review_reason": "Damaged."})
        self.assertEqual(tools.execute("T1", "backend_write", {**REFUND, "amount": 50})["status"], "executed")
        self.assertEqual(len(self.reviews), 2)

    def test_a_write_that_needs_no_review_is_said_so(self):
        tools = self.tools("agent")
        result = tools.execute("T1", "request_review", {**REFUND, "amount": 20, "review_reason": "Damaged."})
        self.assertEqual((result["status"], self.reviews), ("not_required", []))

    def test_rejections_follow_max_rejections_as_with_the_automatic_trigger(self):
        tools = self.tools("agent", {"decision": "reject", "issues": ["Not received"]},
                           {"decision": "reject", "issues": ["Still not received"]})
        first = tools.execute("T1", "request_review", {**REFUND, "amount": 50, "review_reason": "Damaged."})
        self.assertEqual(first["status"], "blocked")   # the agent may fix it once
        second = tools.execute("T1", "request_review", {**REFUND, "amount": 50, "review_reason": "Damaged."})
        self.assertEqual(second["status"], "awaiting_human")
        self.assertEqual(tools.execute("T1", "backend_write", {**REFUND, "amount": 50})["status"], "review_required")

    def test_automatic_trigger_is_unchanged_and_has_no_request_review_tool(self):
        tools = self.tools("automatic")
        self.assertNotIn("request_review", tools.argument_models)
        self.assertEqual(tools.execute("T1", "backend_write", {**REFUND, "amount": 50})["status"], "review_required")
        written = tools.execute("T1", "backend_write", {**REFUND, "amount": 50, "review_reason": "Damaged."})
        self.assertEqual((written["status"], len(self.reviews)), ("executed", 1))

    def test_an_unknown_trigger_fails_at_start(self):
        with self.assertRaisesRegex(ValueError, "review.trigger must be one of automatic, agent"):
            self.tools("sometimes")


class ConfigTests(unittest.TestCase):
    def test_s3_s4_ask_for_the_review_themselves(self):
        for path in ("configs/experiments/multi_agent_rag.yaml", "configs/experiments/multi_agent_skills.yaml"):
            self.assertEqual(load_config(path)["review"]["trigger"], "agent")


if __name__ == "__main__":
    unittest.main()
