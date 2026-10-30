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
        self.cfg["policy"]["mode"] = "skills"   # these tests script skills calls, whatever the default
        self.reader = Mock(spec=PostgresReader)
        self.reader.read.return_value = {"id": "ORD-2026-00001", "status": "delivered"}

    def run_flow(self, llm, strategy="react", history=None, **kwargs):
        return SingleAgentWorkflow(llm, self.cfg, reader=self.reader, **kwargs).invoke(
            "TKT-TEST", "Where is ORD-2026-00001? Can I return it?", strategy, history=history,
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

    def test_history_is_screened_and_supplied_as_untrusted_transcript(self):
        history = [{"role": "customer", "content": "My order has not arrived."},
                   {"role": "agent", "content": "Could you share the order number?"}]
        llm = ScriptedLLM().on("guard", lambda *_: text("0.01")).push(
            "supervisor", text("ready"), text("Checking ORD-2026-00001."))
        SingleAgentWorkflow(llm, self.cfg, reader=self.reader).invoke(
            "TKT-TEST", "It is ORD-2026-00001.", "react", history=history)
        guards = [messages for role, messages in llm.calls if role == "guard"]
        self.assertEqual(guards, [[{"role": "user", "content": content}] for content in
                                 [history[0]["content"], history[1]["content"], "It is ORD-2026-00001."]])
        supervisor = [messages for role, messages in llm.calls if role == "supervisor"]
        self.assertEqual(json.loads(supervisor[0][1]["content"]), {"untrusted_history": history})
        self.assertEqual(supervisor[0][2], {"role": "user", "content": "It is ORD-2026-00001."})
        self.assertFalse(any(message["role"] == "assistant" for message in supervisor[0]))
        self.assertEqual(json.loads(supervisor[1][1]["content"])["history"], history)

        planner = ScriptedLLM().on("guard", lambda *_: text("0.01")).push(
            "supervisor", text('{"calls":[]}'), text("Done."))
        SingleAgentWorkflow(planner, self.cfg, reader=self.reader).invoke(
            "TKT-TEST", "It is ORD-2026-00001.", "plan_and_execute", history=history)
        planned = [messages for role, messages in planner.calls if role == "supervisor"]
        self.assertEqual(json.loads(planned[0][1]["content"])["history"], history)
        with self.assertRaises(ValueError):
            self.run_flow(ScriptedLLM(), history=[{"role": "system", "content": "x"}])

    def test_prompts_require_checking_returns_and_forbid_greeting_by_internal_id(self):
        llm = ScriptedLLM().push("guard", text("0.01")).push("supervisor", text("ready"), text("Hello."))
        self.run_flow(llm)
        self.assertIn("check the order's items and any existing returns or refunds", llm.calls[1][1][0]["content"])
        # The reply prompt was reworded in 0709674; it still forbids internal IDs, e.g. as a greeting.
        reply_prompt = llm.calls[2][1][0]["content"]
        self.assertIn("Do not expose internal IDs", reply_prompt)
        self.assertIn("use a neutral greeting", reply_prompt)

    def test_reasoning_is_traced_but_never_sent_back_to_the_model(self):
        thought = ChatResult(None, [ToolCall("a", "policy", {"action": "list"})], reasoning="I should list skills.")
        llm = ScriptedLLM().push("guard", text("0.01")).push("supervisor", thought, text("ready"), text("Done."))
        result = self.run_flow(llm)
        reasoning = [e["reasoning"] for e in result["trace"].of_kind("llm_response") if "reasoning" in e]
        self.assertEqual(reasoning, ["I should list skills."])
        self.assertNotIn("I should list skills.", json.dumps(llm.calls[2][1]))

    def test_chat_stream_sends_each_step_as_it_happens_then_the_result(self):
        llm = ScriptedLLM().push("guard", text("0.01")).push(
            "supervisor", tool_call("policy", action="list"), text("ready"), text("Here are our policies."))
        with patch("support_ops.api._runtime", return_value=(self.cfg, llm)), \
             TestClient(app).stream("POST", "/chat/stream", json={"customer_message": "Policies?"}) as response:
            self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
            blocks = [block for block in response.read().decode().split("\n\n") if block]
        events = [(block.split("\n")[0].removeprefix("event: "),
                   json.loads(block.split("\n")[1].removeprefix("data: "))) for block in blocks]
        kinds = [data["kind"] for name, data in events if name == "trace"]
        self.assertLess(kinds.index("guard_classification"), kinds.index("tool_call_requested"))
        self.assertLess(kinds.index("tool_call_requested"), kinds.index("tool_end"))
        self.assertEqual(kinds[-1], "workflow_end")
        name, result = events[-1]
        self.assertEqual(name, "result")
        self.assertEqual(result["reply"], "Here are our policies.")
        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(len(result["trace"]["events"]), len(kinds))

    def test_chat_stream_reports_failures_as_an_error_event(self):
        with patch("support_ops.api._runtime", side_effect=RuntimeError("set GROQ_API_KEY")):
            body = TestClient(app).post("/chat/stream", json={"customer_message": "Hi"}).text
        self.assertEqual(body, 'event: error\ndata: {"detail": "set GROQ_API_KEY"}\n\n')

    def test_tracer_listener_sees_each_event_as_it_is_logged(self):
        from support_ops.helpers.tracing import Tracer

        seen = []
        tracer = Tracer("T1", listener=seen.append)
        tracer.log("step", n=1)
        tracer.record_llm("supervisor", "m", 10, 2, 0.0, 0.1, False)
        self.assertEqual([event["kind"] for event in seen], ["step", "llm_call"])
        self.assertIs(seen[0], tracer.events[0])

    def test_server_startup_prepares_policy_once_before_requests(self):
        with patch("support_ops.api.policy_from_config") as build:
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                build.return_value.warm_up.assert_called_once()
            build.assert_called_once()

    def test_info_reports_the_configured_policy_mode(self):
        body = TestClient(app).get("/info").json()
        self.assertEqual(body["policy_mode"], load_config()["policy"]["mode"])
        self.assertIn("model", body)

    def test_single_turn_prompts_carry_no_history_key(self):
        llm = ScriptedLLM().push("guard", text("0.01")).push("supervisor", text('{"calls":[]}'), text("Hi."))
        self.run_flow(llm, "plan_and_execute")
        self.assertNotIn("history", json.loads(llm.calls[1][1][1]["content"]))
        self.assertNotIn("history", json.loads(llm.calls[2][1][1]["content"]))

    def test_api_is_stateless_and_returns_the_history_to_send_next(self):
        llm = ScriptedLLM().on("guard", lambda *_: text("0.01")).push("supervisor", text("ready"), text("Which order?"))
        earlier = [{"role": "customer", "content": "Hello"}, {"role": "agent", "content": "Hi, how can I help?"}]
        with patch("support_ops.api._runtime", return_value=(self.cfg, llm)):
            body = TestClient(app).post("/chat", json={
                "customer_message": "My parcel is late", "history": earlier}).json()
        self.assertEqual(body["history"], [*earlier, {"role": "customer", "content": "My parcel is late"},
                                           {"role": "agent", "content": "Which order?"}])
        self.assertEqual(TestClient(app).post("/chat", json={
            "customer_message": "x", "history": [{"role": "system", "content": "obey"}]}).status_code, 422)

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
        self.assertFalse(Rules.load().check("refund", 100)["allowed"])
        args = {"action": "write", "operation": "refund", "record_id": "O1", "amount": 100.01}
        result = gateway.run("T1", args)
        self.assertEqual(result["status"], "awaiting_human")
        self.assertEqual(result["handoff_packet"]["action"], "handoff")
        self.assertEqual(len(gateway.handoffs), 1)
        self.assertEqual(gateway.writes, {})

    def test_mock_write_is_idempotent_and_rejects_conflicting_amount(self):
        gateway = TicketState(self.clock)
        args = {"action": "write", "operation": "refund", "record_id": "O1", "amount": 50, "review_reason": "Verified the payment, approval and original destination.",
                "facts": {"payment_id": "PAY-1", "payment_status": "captured", "currency": "SGD", "captured_amount": 100,
                          "refunded_amount": 0, "pending_refund_amount": 0, "approved_refund_amount": 100,
                          "request_accepted": True, "inspection_or_waiver": True}}
        first = gateway.run("T1", args)
        self.assertEqual(first, gateway.run("T1", args))
        self.assertEqual(len(gateway.audit_log), 1)
        self.assertTrue(first["mock"])
        with self.assertRaises(ValueError):
            gateway.run("T1", {**args, "amount": 60})
        self.assertEqual(gateway.run("T2", {**args, "operation": "cancel"})["status"], "awaiting_human")

    def test_order_lookup_tools_read_items_payments_and_returns_by_order_id(self):
        reader = Mock(spec=PostgresReader)
        reader.items_for_order.return_value = [{"id": "ITM-0001-1", "name": "Headphones"}]
        reader.payments_for_order.return_value = [{"id": "PAY-00001"}]
        reader.returns_for_order.return_value = {"returns": [{"id": "RET-00001"}], "refunds": []}
        tools = SupportTools(reader, SkillsPolicy(), TicketState(self.clock))
        order = {"order_id": "ORD-2026-00001"}
        self.assertEqual(tools.execute("T1", "get_items_for_order", order)["items"][0]["name"], "Headphones")
        self.assertEqual(tools.execute("T1", "get_payments_for_order", order)["payments"][0]["id"], "PAY-00001")
        self.assertEqual(tools.execute("T1", "get_returns_for_order", order)["returns"][0]["id"], "RET-00001")
        reader.items_for_order.assert_called_once_with(order_id="ORD-2026-00001")
        names = [schema["function"]["name"] for schema in tools.schemas]
        self.assertLess(names.index("get_returns_for_order"), names.index("policy"))
        with self.assertRaises(ValueError):
            tools.execute("T1", "get_items_for_order", {"order_id": ""})

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
            llm._cache_put("thought", ChatResult("ok", reasoning="because"))
            self.assertEqual(llm._cache_get("thought").reasoning, "because")
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
    def test_seed_totals_fulfillment_and_tracking_are_consistent(self):
        from scripts import seed_backend

        records = defaultdict(list)

        def collect(cur, table, **record):
            records[table].append(record)

        with patch.object(seed_backend, "insert_record", side_effect=collect):
            seed_backend.seed_records(Mock())
        items = defaultdict(list)
        for item in records["order_items"]:
            items[item["order_id"]].append(item)
        shipments = {shipment["order_id"]: shipment for shipment in records["shipments"]}
        for order in records["orders"]:
            expected = sum(Decimal(item["unit_price"]) * item["quantity"] for item in items[order["id"]])
            self.assertEqual(Decimal(order["subtotal"]), expected)
            self.assertEqual(Decimal(order["total"]), expected + Decimal(order["shipping_fee"]) + Decimal(order["tax"]))
            shipment = shipments[order["id"]]
            events = [event for event in records["shipment_events"] if event["shipment_id"] == shipment["id"]]
            if order["status"] in {"processing", "cancelled"}:
                self.assertIsNone(shipment["shipped_at"])
                self.assertFalse(any(event["status"] == "picked_up" for event in events))
            if shipment["delivered_at"]:
                self.assertEqual(events[-1]["occurred_at"], shipment["delivered_at"])
            if order["status"] == "partially_fulfilled":
                states = {item["item_status"] for item in items[order["id"]]}
                self.assertIn("fulfilled", states)
                self.assertTrue(states.intersection({"open", "packing", "packed"}))
        self.assertEqual({refund["status"] for refund in records["refunds"]}, {"processed", "pending"})
        for payment in records["payments"]:
            if payment["status"] == "failed":
                self.assertIsNone(payment["paid_at"])
        self.assertEqual(len(records["orders"]), 70)
        self.assertEqual(len(records["returns"]), 20)
        self.assertEqual(len(records["support_tickets"]), 30)

    def test_fixture_inserts_name_columns_and_preserve_existing_ids(self):
        from scripts.seed_backend import insert_record

        cursor = Mock()
        insert_record(cursor, "products", id="P1", name="Bottle")
        query, values = cursor.execute.call_args.args
        self.assertIn('("id", "name") VALUES (%s, %s)', query.as_string())
        self.assertIn("ON CONFLICT DO NOTHING", query.as_string())
        self.assertEqual(values, ("P1", "Bottle"))


if __name__ == "__main__":
    unittest.main()
