"""The signed-in account scopes every read and write, in both architectures."""
import json
import unittest
from unittest.mock import Mock, patch

from support_ops.api import signed_in
from support_ops.backend.accounts import AccountReader, AccountWriter, for_account
from support_ops.backend.postgres import PostgresReader
from support_ops.helpers.config import load_config
from support_ops.helpers.llm import text, tool_call
from support_ops.systems.single_agent import SingleAgentWorkflow
from tests.fake_writer import FakeWriter
from tests.scripted import ScriptedLLM

MINE, THEIRS = "ORD-MINE", "ORD-THEIRS"


def reader() -> Mock:
    backend = Mock(spec=PostgresReader)
    rows = {
        ("orders", MINE): {"id": MINE, "customer_id": "CUS-A", "status": "delivered"},
        ("orders", THEIRS): {"id": THEIRS, "customer_id": "CUS-B", "status": "delivered"},
        ("customers", "CUS-A"): {"id": "CUS-A"}, ("customers", "CUS-B"): {"id": "CUS-B"},
        ("payments", "PAY-MINE"): {"id": "PAY-MINE", "order_id": MINE},
        ("payments", "PAY-THEIRS"): {"id": "PAY-THEIRS", "order_id": THEIRS},
        ("support_tickets", "TKT-B"): {"id": "TKT-B", "customer_id": "CUS-B"},
        ("products", "PRD-1"): {"id": "PRD-1"},
    }
    backend.read.side_effect = lambda kind, record_id: rows.get(
        (kind, record_id), {"not_found": True, "kind": kind, "id": record_id})
    backend.items_for_order.return_value = [{"id": "ITM-1"}]
    backend.payments_for_order.return_value = [{"id": "PAY-1"}]
    backend.shipments_for_order.return_value = [{"id": "SHP-1"}]
    backend.shipment_events_for_order.return_value = [{"id": "EVT-1"}]
    backend.returns_for_order.return_value = {"returns": [{"id": "RET-1"}], "refunds": []}
    return backend


class AccountReaderTests(unittest.TestCase):
    def test_another_customers_records_read_exactly_as_missing_ones(self):
        scoped = AccountReader(reader(), "CUS-A")
        for kind, record_id in (("orders", THEIRS), ("customers", "CUS-B"), ("payments", "PAY-THEIRS"),
                                ("support_tickets", "TKT-B")):
            with self.subTest(kind=kind):
                self.assertEqual(scoped.read(kind, record_id), {"not_found": True, "kind": kind, "id": record_id})
        self.assertEqual(scoped.read("orders", MINE)["id"], MINE)
        self.assertEqual(scoped.read("payments", "PAY-MINE")["id"], "PAY-MINE")
        self.assertEqual(scoped.read("customers", "CUS-A")["id"], "CUS-A")
        self.assertEqual(scoped.read("products", "PRD-1")["id"], "PRD-1")   # the catalogue is public

    def test_order_lookups_return_nothing_for_another_customers_order(self):
        scoped = AccountReader(reader(), "CUS-A")
        self.assertEqual(scoped.items_for_order(THEIRS), [])
        self.assertEqual(scoped.payments_for_order(THEIRS), [])
        self.assertEqual(scoped.shipments_for_order(THEIRS), [])
        self.assertEqual(scoped.shipment_events_for_order(THEIRS), [])
        self.assertEqual(scoped.returns_for_order(THEIRS), {"returns": [], "refunds": []})
        self.assertEqual(scoped.items_for_order(MINE), [{"id": "ITM-1"}])
        self.assertEqual(scoped.returns_for_order(MINE)["returns"], [{"id": "RET-1"}])

    def test_the_writer_passes_the_account_and_signed_out_changes_nothing(self):
        fake = FakeWriter()
        writer = AccountWriter(fake, "CUS-A")
        self.assertEqual(writer.save_ticket(ticket_id="T1", status="resolved")["customer_id"], "CUS-A")
        backend, plain = reader(), FakeWriter()
        self.assertEqual(for_account(backend, plain, None), (backend, plain))
        scoped_reader, scoped_writer = for_account(backend, plain, "CUS-A")
        self.assertIsInstance(scoped_reader, AccountReader)
        self.assertIsInstance(scoped_writer, AccountWriter)

    def test_an_unknown_account_is_refused_before_the_workflow_runs(self):
        self.assertEqual(signed_in("CUS-A", reader()), "CUS-A")
        self.assertIsNone(signed_in(None, reader()))
        with self.assertRaisesRegex(ValueError, "unknown customer account CUS-Z"):
            signed_in("CUS-Z", reader())


class CustomerOrdersToolTests(unittest.TestCase):
    def tools(self, backend, customer_id=None):
        from support_ops.gateway.ticket_state import TicketState
        from support_ops.helpers.clock import Clock
        from support_ops.policy.skills import SkillsPolicy
        from support_ops.systems.tools import SupportTools
        from datetime import datetime, timezone

        scoped, writer = for_account(backend, FakeWriter(), customer_id)
        return SupportTools(scoped, SkillsPolicy(), TicketState(Clock(datetime(2026, 10, 31, tzinfo=timezone.utc))),
                            writer)

    def test_it_lists_only_the_signed_in_customers_orders_and_takes_no_customer_argument(self):
        backend = reader()
        backend.orders_for_customer.return_value = [{"id": MINE, "items": "Mug"}]
        result = self.tools(backend, "CUS-A").execute("T1", "get_orders_for_customer", {})
        self.assertEqual(result, {"orders": [{"id": MINE, "items": "Mug"}]})
        backend.orders_for_customer.assert_called_once_with("CUS-A")
        with self.assertRaises(ValueError):   # the model cannot name another customer
            self.tools(backend, "CUS-A").execute("T1", "get_orders_for_customer", {"customer_id": "CUS-B"})

    def test_signed_out_it_asks_for_the_order_number_instead(self):
        backend = reader()
        with self.assertRaisesRegex(ValueError, "no customer is signed in"):
            self.tools(backend).execute("T1", "get_orders_for_customer", {})
        backend.orders_for_customer.assert_not_called()

    def test_both_specialists_and_the_single_agent_have_it(self):
        from support_ops.agents.toolbox import ORDERS_SHIPPING_TOOLS, RETURNS_REFUNDS_TOOLS

        self.assertIn("get_orders_for_customer", ORDERS_SHIPPING_TOOLS)
        self.assertIn("get_orders_for_customer", RETURNS_REFUNDS_TOOLS)
        names = [schema["function"]["name"] for schema in self.tools(reader()).schemas]
        self.assertIn("get_orders_for_customer", names)


class SignedInWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.cfg["policy"]["mode"] = "skills"

    def test_the_single_agent_sees_only_the_account_and_saves_the_ticket_as_theirs(self):
        llm = ScriptedLLM().push("guard", text("0.01")).push(
            "supervisor", tool_call("read_record", kind="orders", record_id=THEIRS),
            tool_call("update_ticket", status="awaiting_customer", category="delivery",
                      subject="Asked where an order is.", customer_message="Where is ORD-THEIRS?"),
            text("ready"), text("Could you check the order number?"),
        )
        writer = FakeWriter()
        result = SingleAgentWorkflow(llm, self.cfg, reader=reader(), writer=writer, customer_id="CUS-A").invoke(
            "T1", "Where is ORD-THEIRS?", "react")
        self.assertTrue(result["evidence"][0]["result"]["not_found"])
        self.assertEqual(writer.ticket_saves[0]["customer_id"], "CUS-A")
        system = llm.calls[1][1][0]["content"]
        self.assertIn("signed in as account CUS-A", system)
        self.assertIn("call update_ticket", system)
        checked = json.loads([messages for role, messages in llm.calls if role == "approver"][0][1]["content"])
        self.assertEqual(checked["case"]["signed_in_account"], "CUS-A")

    def test_signed_out_the_prompt_names_no_account(self):
        llm = ScriptedLLM().push("guard", text("0.01")).push("supervisor", text("ready"), text("Hello."))
        SingleAgentWorkflow(llm, self.cfg, reader=reader(), writer=FakeWriter()).invoke("T1", "Hi", "react")
        self.assertNotIn("signed in as account", llm.calls[1][1][0]["content"])

    def test_the_multi_agent_workflow_scopes_its_reader_writer_and_brief(self):
        from support_ops.systems.multi_agent import MultiAgentWorkflow

        cfg = load_config("configs/experiments/multi_agent_skills.yaml")
        workflow = MultiAgentWorkflow(ScriptedLLM(), cfg, reader=reader(), writer=FakeWriter(), customer_id="CUS-A")
        self.assertIsInstance(workflow.reader, AccountReader)
        self.assertIsInstance(workflow.writer, AccountWriter)
        brief = workflow._brief({"customer_message": "Hi", "history": []})
        self.assertEqual(brief["signed_in_account"], "CUS-A")

    def test_the_supervisor_design_inherits_the_scoping_and_shared_review_limit(self):
        from support_ops.systems.supervisor_agent import SupervisorAgentWorkflow

        cfg = load_config("configs/experiments/multi_agent_skills.yaml")
        workflow = SupervisorAgentWorkflow(ScriptedLLM(), cfg, reader=reader(), writer=FakeWriter(),
                                           customer_id="CUS-A")
        self.assertIsInstance(workflow.reader, AccountReader)
        self.assertIsInstance(workflow.writer, AccountWriter)
        self.assertEqual(workflow.settings["max_reproposals"], cfg["review"]["max_rejections"])


class ApiTests(unittest.TestCase):
    def test_the_customer_list_and_an_unknown_account_over_http(self):
        from fastapi.testclient import TestClient
        from support_ops.api import app

        client = TestClient(app)
        with patch("support_ops.api.PostgresReader") as backend:
            backend.return_value.customers.return_value = [{"id": "CUS-A", "first_name": "Ana"}]
            self.assertEqual(client.get("/customers").json(), [{"id": "CUS-A", "first_name": "Ana"}])
            backend.return_value.read.return_value = {"not_found": True}
            response = client.post("/chat", json={"customer_message": "Hi", "customer_id": "CUS-Z"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("unknown customer account CUS-Z", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
