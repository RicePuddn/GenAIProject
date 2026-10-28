"""Shared hard rules and the pluggable policy provider that separates S1 (RAG) from S2 (skills)."""
import json
import tempfile
import unittest
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

from support_ops.backend.postgres import PostgresReader
from support_ops.helpers.arguments import NonEmpty, ToolArguments
from support_ops.helpers.config import load_config
from support_ops.helpers.llm import ScriptedLLM, text, tool_call
from support_ops.policy.rules import Rules
from support_ops.systems.single_agent import SingleAgentWorkflow

RULES_YAML = """
version: "test"
currency: SGD
refund:
  rule_id: test_limit
  max_auto_amount: "{limit}"
"""


def write_rules(directory: str, limit: str) -> Path:
    path = Path(directory) / "rules.yaml"
    path.write_text(RULES_YAML.format(limit=limit))
    return path


class RulesTests(unittest.TestCase):
    def test_refund_limit_boundary_comes_from_rules_file(self):
        rules = Rules.load()
        self.assertEqual(rules.refund_limit, 100)
        self.assertEqual(rules.check("refund", 100), {
            "allowed": True, "requires_human": False,
            "rule": "mock_refund_limit_100", "limit": 100.0, "currency": "SGD",
        })
        above = rules.check("refund", 100.01)
        self.assertFalse(above["allowed"])
        self.assertTrue(above["requires_human"])

    def test_unsupported_operations_need_a_human_and_bad_amounts_raise(self):
        rules = Rules.load()
        self.assertTrue(rules.check("cancel", None)["requires_human"])
        for amount in (None, 0, -1, float("nan"), float("inf")):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                rules.check("refund", amount)

    def test_invalid_rules_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for limit in ("0", "-5", "abc"):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    Rules.load(write_rules(directory, limit))
            path = Path(directory) / "missing.yaml"
            path.write_text("version: x\n")
            with self.assertRaises(ValueError):
                Rules.load(path)


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
