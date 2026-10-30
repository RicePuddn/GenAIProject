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
from support_ops.agents.specialist import citation_forms
from support_ops.agents.toolbox import Toolbox
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
        self.policy = SkillsPolicy(skills_dir())
        tools = SupportTools(Mock(), self.policy, Mock())
        self.box = Toolbox(tools, [], {"policy": self.policy.scoped_tool(["returns-and-refunds"])})

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

    def test_unknown_skill_names_the_real_ones(self):
        # A guessed name is not another area, so the error lists the specialist's skills, not scope all.
        with self.assertRaisesRegex(ValueError, r"no policy skill named 'returns'; your skills: \['returns-and-refunds'\]") as error:
            self.call(action="open", skill="returns")
        self.assertNotIn("scope all", str(error.exception))
        with self.assertRaisesRegex(ValueError, "your skills: \\['orders-and-shipping', 'returns-and-refunds'\\]"):
            self.call(action="open", skill="returns", scope="all")
        with self.assertRaisesRegex(ValueError, "outside your policy skills"):   # a real skill of another area
            self.call(action="open", skill="orders-and-shipping")

    def test_planner_sees_only_its_own_skills(self):
        # Comment 2 on PR #5: listing every skill made plans read files the default scope refuses.
        context = self.box.policy.planner_context()
        self.assertIn("returns-and-refunds", context)
        self.assertNotIn("orders-and-shipping", context)

    def test_errors_only_offer_skills_that_exist(self):
        # Comment 4 on PR #5: a mistyped scope was offered back to the model as a valid name.
        tools = SupportTools(Mock(), self.policy, Mock())
        box = Toolbox(tools, [], {"policy": self.policy.scoped_tool(["returns-and-refund"])})
        listed = box.execute("T1", "policy", {"action": "list"})
        self.assertEqual(listed["skills"], [])
        self.assertIn("hand the case over", listed["note"])
        with self.assertRaisesRegex(ValueError, "no policy skill named 'returns-and-refund'; no policy skills are available"):
            box.execute("T1", "policy", {"action": "open", "skill": "returns-and-refund"})

    def test_no_path_escape_and_no_check(self):
        # Out of the skill folder through the file name, e.g. into the other specialist's skill.
        with self.assertRaisesRegex(ValueError, "invalid policy path"):
            self.call(action="read", skill="returns-and-refunds", file="../orders-and-shipping/tracking.md")
        with self.assertRaisesRegex(ValueError, "no policy skill named '..'"):
            self.call(action="read", skill="..", file="secrets.md", scope="all")
        with self.assertRaises(ValidationError):   # the route tool applies the hard rules
            self.call(action="check", operation="refund", amount=5)
        with self.assertRaisesRegex(ValidationError, "read requires file"):
            self.call(action="read", skill="returns-and-refunds")


class SkillsSetupTests(unittest.TestCase):
    """Comments 3 and 4 on PR #5: problems in the skills folder show up at start-up, not mid-ticket."""

    def broken(self) -> Path:
        root = skills_dir()
        (root / "orders-and-shipping" / "SKILL.md").write_text("---\ndescription: bad: yaml\n---\n# Orders\n")
        return root

    def test_one_bad_skill_md_does_not_break_the_others(self):
        catalog = SkillsPolicy(self.broken()).catalog()
        self.assertEqual([item["skill"] for item in catalog], ["orders-and-shipping", "returns-and-refunds"])
        self.assertIn("could not be read", catalog[0]["error"])

    def test_warm_up_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "unreadable SKILL.md front matter in \\['orders-and-shipping'\\]"):
            SkillsPolicy(self.broken()).warm_up()
        with self.assertRaisesRegex(ValueError, "policy skills folder not found"):
            SkillsPolicy("/no/such/skills").warm_up()
        with self.assertRaisesRegex(ValueError, "no policy skills"):
            SkillsPolicy(tempfile.mkdtemp()).warm_up()
        SkillsPolicy(skills_dir()).warm_up()   # a good folder passes


class SkillsBase(Base):
    """The S4 config, run against a temporary skills folder."""
    def setUp(self):
        super().setUp()
        self.cfg = load_config("configs/experiments/multi_agent_skills.yaml")
        self.cfg["multi_agent"]["check_citations"] = False   # as in Base; turned on where a test reads first

    def run_flow(self, message="My mug arrived broken, refund please. ORD-2026-00001", strategy="react", **backend):
        self.workflow = MultiAgentWorkflow(self.llm, self.cfg, reader=reader(**backend),
                                           policy=SkillsPolicy(skills_dir()), clock=Clock(NOW))
        return self.workflow.invoke("T1", message, strategy)



class SkillsWorkflowTests(SkillsBase):
    def specialist_prompt(self):
        return next(messages[0]["content"] for role, messages in self.llm.calls if role == "specialist")

    def test_refund_ticket_reads_skills_and_executes(self):
        self.cfg["multi_agent"]["check_citations"] = True   # it reads refunds.md before citing it
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

    def test_a_bad_skill_md_elsewhere_does_not_crash_the_ticket(self):
        # Comment 3 on PR #5: the returns specialist's ticket used to raise ScannerError.
        root = skills_dir()
        (root / "orders-and-shipping" / "SKILL.md").write_text("---\ndescription: bad: yaml\n---\n# Orders\n")
        self.llm.push("supervisor", triage(), text("Recorded.")).push(
            "specialist", tool_call("policy", action="list"), route()).push("approver", verdict())
        self.workflow = MultiAgentWorkflow(self.llm, self.cfg, reader=reader(), policy=SkillsPolicy(root), clock=Clock(NOW))
        result = self.workflow.invoke("T1", "My mug arrived broken, refund please. ORD-2026-00001", "react")
        self.assertEqual(result["ticket_status"], "resolved")

    def test_a_mistyped_skill_folder_fails_at_start(self):
        # Comment 4 on PR #5: it used to start fine and leave the specialist with empty lists.
        self.cfg["multi_agent"]["policy_skills"] = {"returns_refunds": ["returns-and-refund"]}
        with self.assertRaisesRegex(ValueError, r"policy_skills for returns_refunds .*\['returns-and-refund'\]"):
            self.run_flow()


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


class CitationTests(SkillsBase):
    """Comment 2 on PR #5: a proposal may only cite policy the specialist actually read in this case,
    and a plan whose call failed is made again rather than turned into a proposal."""

    def setUp(self):
        super().setUp()
        self.cfg["multi_agent"]["check_citations"] = True

    @staticmethod
    def draft(*cited):
        return text(json.dumps({"action": "refund", "order_id": "ORD-2026-00001", "amount": 20,
                                "items": [{"item_id": "ITM-0031-2", "quantity": 1}], "cited_clauses": list(cited),
                                "route": "auto", "reason": "Damaged on arrival."}))

    def test_a_refused_planned_read_makes_the_specialist_plan_again(self):
        # Her reproduction: the refused read of tracking.md went straight on to an auto refund citing it.
        other_area = {"calls": [{"name": "policy", "arguments": {
            "action": "read", "skill": "orders-and-shipping", "file": "tracking.md"}}]}
        own_area = {"calls": [{"name": "policy", "arguments": {
            "action": "read", "skill": "returns-and-refunds", "file": "refunds.md"}}]}
        self.llm.push("supervisor", triage(), text("Your refund of 20 SGD is recorded.")).push(
            "specialist", text(json.dumps(other_area)), text(json.dumps(own_area)),
            self.draft("returns-and-refunds/refunds.md")).push("approver", verdict())
        result = self.run_flow(strategy="plan_and_execute")
        rejected = result["trace"].of_kind("plan_rejected")
        self.assertIn("planned call policy failed", rejected[0]["reason"])
        self.assertIn("without repeating calls", rejected[0]["reason"])
        self.assertEqual(result["ticket_status"], "resolved")
        self.assertEqual(result["proposal"]["cited_clauses"], ["returns-and-refunds/refunds.md"])

    def test_citing_a_file_that_was_not_read_is_rejected(self):
        self.llm.push("supervisor", triage(), text("Your refund of 20 SGD is recorded.")).push(
            "specialist", tool_call("policy", action="read", skill="returns-and-refunds", file="refunds.md"),
            route(cited_clauses=["orders-and-shipping/tracking.md"]),   # never read
            route(cited_clauses=["refunds.md"])).push("approver", verdict())
        result = self.run_flow()
        self.assertEqual(result["trace"].of_kind("citation_rejected")[0]["unread"], ["orders-and-shipping/tracking.md"])
        self.assertIn("were not returned by your policy tool", result["evidence"][1]["result"]["error"])
        self.assertEqual((result["ticket_status"], result["proposal"]["cited_clauses"]), ("resolved", ["refunds.md"]))

    def test_rag_citations_are_the_clause_ids_a_search_returned(self):
        self.llm.push("supervisor", triage(), text("Your refund of 20 SGD is recorded.")).push(
            "specialist", tool_call("policy", action="search", query="damaged item refund"),
            route(cited_clauses=["REF-9"]), route(cited_clauses=["REF-1"])).push("approver", verdict())
        self.workflow = MultiAgentWorkflow(self.llm, load_config("configs/experiments/multi_agent_rag.yaml"),
                                           reader=reader(), policy=rag_policy(), clock=Clock(NOW))
        result = self.workflow.invoke("T1", "My mug arrived broken, refund please. ORD-2026-00001", "react")
        self.assertEqual(result["trace"].of_kind("citation_rejected")[0]["unread"], ["REF-9"])
        self.assertEqual(result["ticket_status"], "resolved")

    def test_a_clause_id_heading_in_a_read_file_can_be_cited(self):
        self.assertEqual(SkillsPolicy.citable({"skill": "returns-and-refunds", "file": "refunds.md",
                                               "content": "# Refunds\n## RET-1 Return window\nWithin 30 days."}),
                         {"refunds.md", "returns-and-refunds/refunds.md", "RET-1"})

    def test_a_read_file_can_be_cited_by_case_path_or_anchor(self):
        self.llm.push("supervisor", triage(), text("Your refund of 20 SGD is recorded.")).push(
            "specialist", tool_call("policy", action="read", skill="returns-and-refunds", file="refunds.md"),
            route(cited_clauses=["Returns-And-Refunds/Refunds.md", "policy/skills/returns-and-refunds/refunds.md",
                                 "refunds.md#damaged-items"])).push("approver", verdict())
        result = self.run_flow()
        self.assertEqual(result["trace"].of_kind("citation_rejected"), [])
        self.assertEqual(result["ticket_status"], "resolved")

    def test_the_same_file_name_in_another_skill_is_still_unread(self):
        self.llm.push("supervisor", triage(), text("Your refund of 20 SGD is recorded.")).push(
            "specialist", tool_call("policy", action="open", skill="returns-and-refunds"),
            route(cited_clauses=["policy/skills/orders-and-shipping/SKILL.md"]),   # opened the other SKILL.md
            route(cited_clauses=["returns-and-refunds/SKILL.md"])).push("approver", verdict())
        result = self.run_flow()
        self.assertEqual(result["trace"].of_kind("citation_rejected")[0]["unread"],
                         ["policy/skills/orders-and-shipping/SKILL.md"])
        self.assertEqual(result["ticket_status"], "resolved")

    def test_citation_forms(self):
        self.assertIn("ret-1", citation_forms("RET-1 Return window"))
        self.assertIn("ret-1", citation_forms("refunds.md#RET-1"))
        self.assertIn("returns-and-refunds/refunds.md", citation_forms("/policy/skills/returns-and-refunds/refunds.md"))
        self.assertNotIn("refunds.md", citation_forms("orders-and-shipping/refunds.md"))


class ScopeSettingTests(SkillsBase):
    def test_a_partial_scope_map_keeps_the_other_specialists_default(self):
        self.cfg["multi_agent"]["policy_skills"] = {"returns_refunds": ["returns-and-refunds", "orders-and-shipping"]}
        self.cfg["multi_agent"]["policy_sections"] = {"orders_shipping": ["Orders"]}
        workflow = MultiAgentWorkflow(self.llm, self.cfg, reader=reader(), policy=SkillsPolicy(skills_dir()),
                                      clock=Clock(NOW))
        self.assertEqual(workflow.settings["policy_skills"],
                         {"orders_shipping": ["orders-and-shipping"],
                          "returns_refunds": ["returns-and-refunds", "orders-and-shipping"]})
        self.assertEqual(workflow.settings["policy_sections"],
                         {"orders_shipping": ["Orders"], "returns_refunds": ["Returns and refunds"]})

    def test_an_explicit_null_scope_still_means_the_whole_policy(self):
        self.cfg["multi_agent"]["policy_skills"] = {"orders_shipping": None}
        workflow = MultiAgentWorkflow(self.llm, self.cfg, reader=reader(), policy=SkillsPolicy(skills_dir()),
                                      clock=Clock(NOW))
        self.assertIsNone(workflow.settings["policy_skills"]["orders_shipping"])

    def test_skills_scope_is_checked_once_at_start_not_per_ticket(self):
        self.llm.push("supervisor", triage(), text("Recorded.")).push("specialist", route()).push("approver", verdict())
        with patch.object(SkillsPolicy, "check_scope", autospec=True, return_value=[]) as check_scope:
            self.run_flow()
        self.assertEqual(check_scope.call_count, len(self.workflow.settings["policy_skills"]))


class SkillMdCitationTests(unittest.TestCase):
    def test_skill_md_must_name_its_skill(self):
        citable = SkillsPolicy.citable({"skill": "returns-and-refunds", "file": "SKILL.md", "content": "# Returns\n"})
        self.assertEqual(citable, {"returns-and-refunds/SKILL.md"})
