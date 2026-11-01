"""The signed-in account scopes every read and write, the same way in every setup."""
import unittest
from unittest.mock import Mock, patch

from support_ops.api import signed_in
from support_ops.setups.multi_common import SPECIALIST_TOOLS
from support_ops.shared.db import PostgresReader
from support_ops.shared.db.accounts import AccountReader, AccountWriter, for_account
from support_ops.shared.tools import ALL_TOOLS
from tests.fakes import CUSTOMER, ORDER, FakeDatabase, ScriptedLLM, services, text, tool_call
from tests.setups.scripts import run

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
        fake = FakeDatabase()
        writer = AccountWriter(fake, "CUS-A")
        self.assertEqual(writer.save_ticket(ticket_id="T1", status="resolved")["customer_id"], "CUS-A")
        backend, plain = reader(), FakeDatabase()
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
        scoped, writer = for_account(backend, FakeDatabase(), customer_id)
        return services(scoped, writer=writer)

    def test_it_lists_only_the_signed_in_customers_orders_and_takes_no_customer_argument(self):
        backend = reader()
        backend.orders_for_customer.return_value = [{"id": MINE, "items": "Mug"}]
        self.assertEqual(self.tools(backend, "CUS-A").call("get_orders_for_customer"),
                         {"orders": [{"id": MINE, "items": "Mug"}]})
        backend.orders_for_customer.assert_called_once_with("CUS-A")
        refused = self.tools(backend, "CUS-A").call("get_orders_for_customer", customer_id="CUS-B")
        self.assertIn("Extra inputs are not permitted", refused["error"])   # the model cannot name another customer

    def test_signed_out_it_asks_for_the_order_number_instead(self):
        backend = reader()
        self.assertIn("no customer is signed in", self.tools(backend).call("get_orders_for_customer")["error"])
        backend.orders_for_customer.assert_not_called()

    def test_both_specialists_and_the_single_agent_have_it(self):
        for tools in (*SPECIALIST_TOOLS.values(), ALL_TOOLS):
            self.assertIn("get_orders_for_customer", tools)


class SignedInSetupTests(unittest.TestCase):
    def test_every_setup_reads_writes_and_saves_as_the_account(self):
        for setup_id, role in (("s1-a", "agent"), ("s5-a", "agent"), ("s3-a", "specialist"), ("s8-a", "specialist")):
            with self.subTest(setup=setup_id):
                db = FakeDatabase()
                db.tables["orders"][ORDER]["customer_id"] = "CUS-2"   # another customer's order
                llm = ScriptedLLM().push(role, tool_call("read_record", kind="orders", record_id=ORDER))
                if role == "specialist":
                    llm.push("supervisor", tool_call("ask_specialist", specialist="orders_shipping", request="Find it."))
                agentic = setup_id in {"s1-a", "s3-a"}
                llm.push(role, tool_call("finish", outcome="awaiting_customer", note="Check the number?") if agentic
                         else text("Gathered."))
                if not agentic:
                    llm.push(role, text('{"outcome": "awaiting_customer", "note": "Check the number?"}'))
                llm.push("supervisor", tool_call("finish", outcome="awaiting_customer", note="Check the number?")
                         if setup_id == "s3-a" else text("Done."))
                output, db = run(setup_id, llm, "Where is ORD-2026-00001?", db=db, customer_id=CUSTOMER)
                self.assertTrue(output.evidence[0]["result"]["not_found"])
                self.assertEqual(db.ticket_saves[-1]["customer_id"], CUSTOMER)
                self.assertEqual(llm.user_payload("approver")["case"]["signed_in_account"], CUSTOMER)

    def test_signed_out_the_prompt_names_no_account(self):
        llm = ScriptedLLM().push("agent", tool_call("finish", outcome="resolved", note="Hello."))
        run("s1-a", llm, "Hi")
        self.assertNotIn("signed in as account", [m for role, m, _ in llm.calls if role == "agent"][0][0]["content"])


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
