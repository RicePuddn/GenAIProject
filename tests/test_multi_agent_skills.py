"""S4: the multi-agent workflow with skills policy, using scripted agents and a temporary skills folder."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from pydantic import ValidationError

from support_ops.api_multi import app
from support_ops.agents.toolbox import Toolbox, policy_skills_tool
from support_ops.helpers.clock import Clock
from support_ops.helpers.config import load_config
from support_ops.helpers.llm import text, tool_call
from support_ops.policy.skills import SkillsPolicy
from support_ops.systems.multi_agent import MultiAgentWorkflow
from support_ops.systems.tools import SupportTools
from tests.test_multi_agent import NOW, Base, rag_policy, reader, route, triage, verdict

SKILLS = {
    "orders-and-shipping": {"SKILL.md": "---\ndescription: Orders.\n---\n# Orders\n- tracking.md\n",
                            "tracking.md": "# Tracking\nShare only the shipment status.\n"},
    "returns-and-refunds": {"SKILL.md": "---\ndescription: Returns.\n---\n# Returns\n- refunds.md\n",
                            "refunds.md": "# Refunds\nRefund damaged items to the original payment.\n"},
}


def skills_dir() -> Path:
    root = Path(tempfile.mkdtemp())
    for skill, files in SKILLS.items():
        (root / skill).mkdir()
        for name, content in files.items():
            (root / skill / name).write_text(content)
    return root


class SkillsToolTests(unittest.TestCase):
    """The returns specialist's skills tool, limited to its own folder by default."""
    def setUp(self):
        tools = SupportTools(Mock(), SkillsPolicy(skills_dir()), Mock())
        self.box = Toolbox(tools, [], {"policy": policy_skills_tool(tools, ["returns-and-refunds"])})

    def call(self, **args):
        return self.box.execute("T1", "policy", args)

    def test_own_skills_by_default_and_all_on_request(self):
        self.assertEqual(self.call(action="list")["skills"], ["returns-and-refunds"])
        self.assertEqual(self.call(action="list", scope="all")["skills"], ["orders-and-shipping", "returns-and-refunds"])
        self.assertIn("original payment", self.call(action="read", skill="returns-and-refunds", file="refunds.md")["content"])
        with self.assertRaisesRegex(ValueError, "outside your policy skills"):
            self.call(action="open", skill="orders-and-shipping")
        self.assertIn("shipment status",
                      self.call(action="read", skill="orders-and-shipping", file="tracking.md", scope="all")["content"])

    def test_no_path_escape_and_no_check(self):
        with self.assertRaisesRegex(ValueError, "invalid policy path"):
            self.call(action="read", skill="..", file="secrets.md", scope="all")
        with self.assertRaises(ValidationError):   # the route tool applies the hard rules
            self.call(action="check", operation="refund", amount=5)
        with self.assertRaisesRegex(ValidationError, "read requires file"):
            self.call(action="read", skill="returns-and-refunds")


class SkillsWorkflowTests(Base):
    def setUp(self):
        super().setUp()
        self.cfg = load_config("configs/experiments/multi_agent_skills.yaml")

    def run_flow(self, message="My mug arrived broken, refund please. ORD-2026-00001", strategy="react", **backend):
        self.workflow = MultiAgentWorkflow(self.llm, self.cfg, reader=reader(**backend),
                                           policy=SkillsPolicy(skills_dir()), clock=Clock(NOW))
        return self.workflow.invoke("T1", message, strategy)

    def specialist_prompt(self):
        return next(messages[0]["content"] for role, messages in self.llm.calls if role == "specialist")

    def test_refund_ticket_reads_skills_and_executes(self):
        self.llm.push("supervisor", triage(), text("Your refund of 20 SGD is recorded.")).push(
            "specialist", tool_call("policy", action="list"),
            tool_call("policy", action="open", skill="returns-and-refunds"),
            tool_call("policy", action="read", skill="returns-and-refunds", file="refunds.md"),
            route(cited_clauses=["returns-and-refunds/refunds.md"])).push("approver", verdict())
        result = self.run_flow()
        reads = [item["result"] for item in result["evidence"] if item["tool"] == "policy"]
        self.assertEqual(reads[0]["skills"], ["returns-and-refunds"])
        self.assertIn("original payment", reads[2]["content"])
        self.assertEqual((result["execution"]["status"], result["ticket_status"]), ("executed", "resolved"))

    def test_prompt_has_the_skills_scope_sentence(self):
        self.llm.push("supervisor", triage(), text("Recorded.")).push("specialist", route()).push("approver", verdict())
        self.run_flow()
        self.assertIn("Your policy skills cover your own area", self.specialist_prompt())
        self.assertNotIn("Policy search", self.specialist_prompt())

    def test_missing_skill_folder_is_logged(self):
        self.cfg["multi_agent"]["policy_skills"] = {"returns_refunds": ["returns-and-refund"]}
        self.llm.push("supervisor", triage(), text("Recorded.")).push("specialist", route()).push("approver", verdict())
        missing = self.run_flow()["trace"].of_kind("policy_skills_missing")
        self.assertEqual(missing[0]["missing"], ["returns-and-refund"])
        self.assertIn("returns-and-refunds", missing[0]["available"])


class RagUnchangedTests(Base):
    def test_rag_prompt_keeps_the_s3_sentence(self):
        self.llm.push("supervisor", triage(), text("Recorded.")).push("specialist", route()).push("approver", verdict())
        self.run_flow()
        prompt = next(messages[0]["content"] for role, messages in self.llm.calls if role == "specialist")
        self.assertIn("Policy search covers your own policy sections; search with scope all when the case "
                      "crosses into another area.", prompt)
        self.assertIsInstance(self.workflow.tools.policy, type(rag_policy()))


class ConfigTests(unittest.TestCase):
    def test_s3_and_s4_configs_differ_only_in_policy(self):
        rag = load_config("configs/experiments/multi_agent_rag.yaml")
        skills = load_config("configs/experiments/multi_agent_skills.yaml")
        self.assertEqual((rag["policy"].pop("mode"), skills["policy"].pop("mode")), ("rag", "skills"))
        rag["multi_agent"].pop("policy_sections")
        skills["multi_agent"].pop("policy_skills")
        self.assertEqual(json.dumps(rag, sort_keys=True, default=str), json.dumps(skills, sort_keys=True, default=str),
                         "multi_agent_rag.yaml and multi_agent_skills.yaml should change together")


class ApiTests(unittest.TestCase):
    def test_api_starts_in_skills_mode_without_the_rag_extra(self):
        # Startup used to warm up RAG whatever the mode, which needs sentence-transformers.
        with patch.dict(os.environ, {"SUPPORT_OPS_MULTI_CONFIG": "configs/experiments/multi_agent_skills.yaml"}), \
                TestClient(app) as client:
            info = client.get("/info").json()
        self.assertEqual((info["policy_mode"], info["corpus"]), ("skills", "policy/skills"))
