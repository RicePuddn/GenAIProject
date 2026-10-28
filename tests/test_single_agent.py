"""Regression checks for strategy routing, tool contracts, approvals and traces."""
import json
import tempfile
import unittest
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from support_ops.api import app
from support_ops.backend.postgres import PostgresReader
from support_ops.gateway.ticket_state import TicketState
from support_ops.helpers.clock import Clock
from support_ops.helpers.config import load_config
from support_ops.helpers.llm import ChatResult, OpenAICompatLLM, ScriptedLLM, ToolCall, text, tool_call
from support_ops.policy.rules import Rules
from support_ops.policy.skills import SkillsPolicy
from support_ops.systems.single_agent import SingleAgentWorkflow
from support_ops.systems.tools import SupportTools


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.reader = Mock(spec=PostgresReader)
        self.reader.read.return_value = {"id": "ORD-2026-00001", "status": "delivered"}

    def run_flow(self, llm, strategy="react", **kwargs):
        return SingleAgentWorkflow(llm, self.cfg, reader=self.reader, **kwargs).invoke(
            "TKT-TEST", "Where is ORD-2026-00001? Can I return it?", strategy,
        )

    def test_react_answers_each_assistant_call_and_replans_after_one_tool(self):
        llm = ScriptedLLM().push("guard", text("0.001")).push(
            "supervisor",
            ChatResult(None, [
                ToolCall("a", "read_record", {"kind": "orders", "record_id": "ORD-2026-00001"}),
                ToolCall("b", "get_shipments_for_order", {"order_id": "ORD-2026-00001"}),
            ]),
            text("ready"), text("Delivered."),
        )
        result = self.run_flow(llm)
        messages = llm.calls[2][1]
        self.assertEqual(len(messages[2]["tool_calls"]), 1)
        self.assertEqual(messages[2]["tool_calls"][0]["id"], messages[3]["tool_call_id"])
        self.reader.shipments_for_order.assert_not_called()
        self.assertEqual(result["trace"].tool_call_count, 1)
        self.assertIsNotNone(result["trace"].tool_calls[0]["execution"])
        self.assertGreater(result["trace"].tool_calls[0]["execution"]["elapsed_s"], 0)

    def test_invalid_plan_retries_without_executing_unknown_tools(self):
        llm = ScriptedLLM().push("guard", text("safe")).push(
            "supervisor", text('{"calls":[{"name":"order_status","arguments":{}}]}'),
            text('{"calls":[{"name":"read_record","arguments":{"kind":"orders","record_id":"ORD-2026-00001"}}]}'),
            text("Delivered."),
        )
        result = self.run_flow(llm, "plan_and_execute")
        self.assertEqual(result["reply"], "Delivered.")
        self.assertEqual(result["replans"], 1)
        self.assertEqual(result["trace"].tool_call_count, 1)
        self.reader.read.assert_called_once()
        self.assertIn("Current workflow date: 2026-10-01", llm.calls[1][1][0]["content"])

    def test_invalid_json_is_a_failed_plan_rather_than_an_empty_plan(self):
        llm = ScriptedLLM().push("guard", text("safe")).push(
            "supervisor", text("not json"), text("{}"), text('{"calls":"wrong"}'),
        )
        result = self.run_flow(llm, "plan_and_execute")
        self.assertEqual(result["ticket_status"], "awaiting_human")
        self.assertEqual(result["trace"].tool_call_count, 0)
        self.assertEqual(result["trace"].llm_calls, 4)

    def test_plan_recovery_receives_completed_evidence_and_failure(self):
        first = {"calls": [
            {"name": "read_record", "arguments": {"kind": "orders", "record_id": "ORD-2026-00001"}},
            {"name": "policy", "arguments": {"action": "read", "skill": "returns-and-refunds", "file": "missing.md"}},
        ]}
        second = {"calls": [
            {"name": "policy", "arguments": {"action": "open", "skill": "returns-and-refunds"}},
            {"name": "policy", "arguments": {"action": "read", "skill": "returns-and-refunds", "file": "eligibility.md"}},
        ]}
        llm = ScriptedLLM().push("guard", text("0.01")).push(
            "supervisor", text(json.dumps(first)), text(json.dumps(second)), text("Policy read."),
        )
        result = self.run_flow(llm, "plan_and_execute")
        correction = json.loads(llm.calls[2][1][1]["content"])
        self.assertEqual(len(correction["evidence"]), 2)
        self.assertIn("policy file not found", correction["previous_error"])
        self.assertEqual(result["trace"].tool_call_count, 4)
        self.assertEqual(len({call["call_id"] for call in result["trace"].tool_calls}), 4)
        self.assertTrue(all(call["execution"] for call in result["trace"].tool_calls))
        self.reader.read.assert_called_once()

    def test_react_tool_loop_hands_off_with_audit_instead_of_graph_recursion_error(self):
        self.cfg["limits"]["max_agent_steps"] = 2
        llm = ScriptedLLM().push("guard", text("safe")).push(
            "supervisor", tool_call("policy", action="list"), tool_call("policy", action="list"),
        )
        gateway = TicketState(Clock(datetime(2026, 10, 1, tzinfo=timezone.utc)))
        result = self.run_flow(llm, ticket_state=gateway)
        self.assertEqual(result["ticket_status"], "awaiting_human")
        self.assertIn("agent step limit", result["handoff_packet"]["reason"])
        self.assertEqual(gateway.audit_log[-1]["event"], "handoff")

    def test_flagged_and_invalid_guard_results_stop_before_supervisor(self):
        for score in ("0.99", "NaN", "", "not safe", "1.2"):
            with self.subTest(score=score):
                llm = ScriptedLLM().push("guard", text(score))
                result = self.run_flow(llm)
                self.assertEqual(result["ticket_status"], "awaiting_human")
                self.assertEqual(len(llm.calls), 1)
                self.assertEqual(result["trace"].tool_call_count, 0)

    def test_api_exposes_matching_tool_count_and_uses_current_date(self):
        llm = ScriptedLLM().push("guard", text("0.01")).push(
            "supervisor", tool_call("policy", action="list"), text("ready"), text("How can I help?"),
        )
        with patch("support_ops.api._runtime", return_value=(self.cfg, llm)):
            response = TestClient(app).post("/chat", json={"customer_message": "Returns policy?", "strategy": "react"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["tool_call_count"], 1)
        self.assertEqual(body["trace"]["tool_call_count"], 1)
        self.assertEqual(len(body["tool_calls"]), 1)
        self.assertIn(str(datetime.now(timezone.utc).date()), llm.calls[1][1][0]["content"])
        self.assertEqual(TestClient(app).post("/chat", json={"customer_message": "   "}).status_code, 422)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock(datetime(2026, 10, 1, tzinfo=timezone.utc))

    def test_policy_entrypoint_aliases_and_clause_discovery(self):
        policy = SkillsPolicy()
        for args in (
            {"action": "read", "file": "returns-and-refunds"},
            {"action": "read", "skill": "returns-and-refunds", "file": "policy.md"},
            {"action": "open", "skill": "returns-and-refunds"},
        ):
            self.assertEqual(policy.run(args)["file"], "SKILL.md")
        catalog = policy.run({"action": "list"})["catalog"]
        self.assertIn("eligibility.md", next(item for item in catalog if item["skill"] == "returns-and-refunds")["files"])
        with self.assertRaises(FileNotFoundError) as error:
            policy.run({"action": "read", "skill": "returns-and-refunds", "file": "missing.md"})
        self.assertIn("eligibility.md", str(error.exception))

    def test_policy_paths_cannot_escape_selected_skill(self):
        policy = SkillsPolicy()
        for skill, filename in (("../../src", "x.md"), ("returns-and-refunds", "../orders-and-shipping/tracking.md")):
            with self.assertRaises(ValueError):
                policy.run({"action": "read", "skill": skill, "file": filename})

    def test_refund_limit_shared_and_above_limit_handoff_does_not_recurse(self):
        gateway = TicketState(self.clock)
        self.assertTrue(Rules.load().check("refund", 100)["allowed"])
        args = {"action": "write", "operation": "refund", "record_id": "O1", "amount": 100.01}
        result = gateway.run("T1", args)
        self.assertEqual(result["status"], "awaiting_human")
        self.assertEqual(result["handoff_packet"]["action"], "handoff")
        self.assertEqual(len(gateway.handoffs), 1)
        self.assertEqual(gateway.writes, {})

    def test_mock_write_is_idempotent_and_rejects_conflicting_amount(self):
        gateway = TicketState(self.clock)
        args = {"action": "write", "operation": "refund", "record_id": "O1", "amount": 50}
        first = gateway.run("T1", args)
        self.assertEqual(first, gateway.run("T1", args))
        self.assertEqual(len(gateway.audit_log), 1)
        self.assertTrue(first["mock"])
        with self.assertRaises(ValueError):
            gateway.run("T1", {**args, "amount": 60})
        self.assertEqual(gateway.run("T2", {**args, "operation": "cancel"})["status"], "awaiting_human")

    def test_tool_contracts_reject_bad_types_enums_missing_fields_and_nonfinite_amounts(self):
        tools = SupportTools(Mock(spec=PostgresReader), SkillsPolicy(), TicketState(self.clock))
        for name, args in (
            ("policy", {"action": "guess"}), ("policy", {"action": "read"}),
            ("read_record", {"kind": "shipments", "record_id": "O1"}),
            ("read_record", {"kind": "orders", "record_id": 5}),
            ("ticket_state", {"action": "write", "operation": "refund", "record_id": "O1"}),
            ("ticket_state", {"action": "write", "operation": "refund", "record_id": "O1", "amount": float("nan")}),
            ("policy", {"action": "check", "operation": "refund", "amount": -1}),
        ):
            with self.subTest(name=name, args=args), self.assertRaises(ValueError):
                tools.validate(name, args)

    def test_cache_ignores_empty_or_corrupt_entries_and_cached_calls_have_no_api_cost(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            cfg = load_config(overrides={"llm.cache_dir": directory})
            llm = OpenAICompatLLM(cfg)
            llm._cache_put("empty", ChatResult(""))
            self.assertFalse((Path(directory) / "empty.json").exists())
            (Path(directory) / "bad.json").write_text("{")
            self.assertIsNone(llm._cache_get("bad"))
            from support_ops.helpers.tracing import Tracer
            llm.client = Mock()
            with patch.object(llm, "_cache_get", return_value=ChatResult("cached", input_tokens=100, output_tokens=50)):
                tracer = Tracer("T1")
                llm.chat("supervisor", [], tracer=tracer)
            self.assertEqual(tracer.cost_usd, 0)
            self.assertTrue(tracer.events[-1]["cached"])
            llm.client.chat.completions.create.assert_not_called()


class SeedTests(unittest.TestCase):
    def test_generated_totals_fulfillment_and_tracking_are_consistent(self):
        from scripts import seed_backend

        records = defaultdict(list)

        def collect(cur, table, **record):
            records[table].append(record)

        with patch.object(seed_backend, "insert_record", side_effect=collect):
            seed_backend.seed_orders(Mock())
            seed_backend.seed_tickets(Mock())
        items = defaultdict(list)
        for item in records["order_items"]:
            items[item["order_id"]].append(item)
        shipments = {shipment["order_id"]: shipment for shipment in records["shipments"]}
        for order in records["orders"]:
            expected = sum(Decimal(item["unit_price"]) * item["quantity"] for item in items[order["id"]])
            self.assertEqual(order["subtotal"], expected)
            self.assertEqual(order["total"], expected + order["shipping_fee"] + order["tax"])
            shipment = shipments[order["id"]]
            events = [event for event in records["shipment_events"] if event["shipment_id"] == shipment["id"]]
            if order["status"] in {"processing", "cancelled"}:
                self.assertIsNone(shipment["shipped_at"])
                self.assertFalse(any(event["status"] == "picked_up" for event in events))
            if shipment["delivered_at"]:
                self.assertEqual(events[-1]["occurred_at"], shipment["delivered_at"])
            if order["status"] == "partially_fulfilled":
                self.assertEqual({item["item_status"] for item in items[order["id"]]}, {"fulfilled", "open"})
        self.assertEqual({refund["status"] for refund in records["refunds"]}, {"processed", "pending"})
        for payment in records["payments"]:
            if payment["status"] == "failed":
                self.assertIsNone(payment["paid_at"])
        self.assertEqual(len(records["orders"]), 60)
        self.assertEqual(len(records["returns"]), 12)
        self.assertEqual(len(records["support_tickets"]), 20)

    def test_fixture_inserts_name_columns_and_preserve_existing_ids(self):
        from scripts.seed_backend import insert_record

        cursor = Mock()
        insert_record(cursor, "products", id="P1", name="Bottle")
        query, values = cursor.execute.call_args.args
        self.assertIn('("id", "name") VALUES (%s, %s)', query.as_string())
        self.assertIn("ON CONFLICT (id) DO NOTHING", query.as_string())
        self.assertEqual(values, ("P1", "Bottle"))


if __name__ == "__main__":
    unittest.main()
