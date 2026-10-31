"""The multi-agent Streamlit page, driven by AppTest against the real API with scripted agents.

The page is served the actual /info and /chat/stream output of `api_multi` for a scripted ticket,
so every event kind the workflow emits is drawn, not a hand-written sample.
"""
import importlib.util
import io
import json
import unittest
from datetime import datetime, timedelta, timezone
from functools import partial
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from support_ops.api_multi import app as api
from support_ops.backend.postgres import PostgresReader
from support_ops.helpers.config import ROOT, load_config
from support_ops.helpers.llm import ScriptedLLM, text, tool_call
from support_ops.policy.rag import RagPolicy
from support_ops.systems.multi_agent import MultiAgentWorkflow
from tests.fake_writer import FakeWriter

ORDER = "ORD-2026-00031"
CLAUSE = {"id": "REF-1", "title": "Refund method and approval", "section": "Returns and refunds",
          "source": "returns-and-refunds.md", "text": "Refunds above the limit need review.", "score": 2.0,
          "via": "search"}


def reader():
    mock = Mock(spec=PostgresReader)
    mock.read.return_value = {"id": ORDER, "total": 275.77, "currency": "SGD", "status": "delivered"}
    now = datetime.now(timezone.utc)
    mock.returns_for_order.return_value = {"returns": [{
        "id": "RET-DEMO", "status": "approved", "reason": "damaged", "dispute_status": "none",
        "requested_at": now - timedelta(days=2), "approved_at": now - timedelta(days=1),
        "reviewed_at": now - timedelta(days=1), "inspection_status": "passed", "evidence_status": "verified",
        "items": [{"order_item_id": "ITM-2", "quantity": 1, "approved_refund_amount": "32.00"}],
    }], "refunds": []}
    mock.payments_for_order.return_value = [{"id": "PAY-DEMO", "amount": "275.77", "status": "captured"}]
    mock.items_for_order.return_value = [{"id": "ITM-2", "quantity": 2, "unit_price": "32.00", "paid_amount": "64.00", "final_sale": False, "product_type": "physical"}]
    mock.shipments_for_order.return_value = [{"id": "SHP-1", "delivered_at": datetime.now(timezone.utc) - timedelta(days=3)}]
    return mock


def scripted_turn(llm: ScriptedLLM, reply: str) -> None:
    """A ticket that touches every step: the supervisor, specialist tools, approval, the write and the reply check."""
    route = tool_call("route", action="refund", order_id=ORDER, amount=32, reason_code="damaged_item",
                      items=[{"item_id": "ITM-2", "quantity": 1}], cited_clauses=["REF-1"], route="approver",
                      reason="One bottle arrived dented.")
    llm.push("guard", text("0.01"), text("0.01"), text("0.01")).push(
        "supervisor",
        text(json.dumps({"thought": "They want a refund for one bottle.", "action": "delegate", "reason": "Refund.",
                         "tasks": [{"specialist": "returns_refunds", "request": "Refund the dented bottle in " + ORDER}],
                         "case": {"issue": "Dented bottle.", "intent": "refund_request", "urgency": "normal",
                                  "language": "en"}})),
        text(json.dumps({"thought": "The refund is done.", "action": "finish", "reason": "Done."})),
        text(reply),
    ).push(
        "specialist",
        tool_call("policy", query="damaged item refund"),
        tool_call("get_items_for_order", order_id=ORDER),
        tool_call("get_returns_for_order", order_id=ORDER),
        route,
    ).push("approver", *[text(json.dumps({"decision": "approve", "issues": [], "reason": "Within policy."}))] * 2)


class RealApi:
    """Answers the page's urllib requests with the multi-agent API's own responses."""

    def __init__(self):
        self.cfg = load_config("configs/experiments/multi_agent_rag.yaml")
        self.llm, self.chat_payloads = ScriptedLLM(), []
        retriever = Mock(search=Mock(return_value=[CLAUSE]), sections=Mock(return_value=["Returns"]))
        self.workflow = partial(MultiAgentWorkflow, writer=FakeWriter(), reader=reader(), policy=RagPolicy(retriever))

    def __call__(self, request, timeout=None):
        with patch("support_ops.api_multi._runtime", return_value=(self.cfg, self.llm)), \
             patch("support_ops.api_multi.MultiAgentWorkflow", self.workflow):
            client = TestClient(api)
            if request.full_url.endswith("/info"):
                return io.BytesIO(client.get("/info").content)
            payload = json.loads(request.data)
            self.chat_payloads.append(payload)
            scripted_turn(self.llm, f"Reply {len(self.chat_payloads)}")
            return io.BytesIO(client.post("/chat/stream", json=payload).content)


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "demo extra not installed")
class DemoMultiAgentTests(unittest.TestCase):
    def test_chat_shows_each_agent_step_and_sends_history_on_the_next_turn(self):
        from streamlit.testing.v1 import AppTest

        fake = RealApi()
        with patch("urllib.request.urlopen", fake):
            app = AppTest.from_file(str(ROOT / "scripts/demo_multi_agent.py"), default_timeout=30).run()
            self.assertIn("rag", app.sidebar.markdown[0].value)
            app.sidebar.radio[0].set_value("plan_and_execute").run()
            app.sidebar.radio[0].set_value("react").run()
            app.chat_input[0].set_value("One bottle in ORD-2026-00031 arrived dented.").run()
            app.chat_input[0].set_value("Thanks!").run()

        self.assertFalse(app.exception)
        self.assertEqual(fake.chat_payloads[0]["strategy"], "react")
        self.assertEqual([turn["content"] for turn in fake.chat_payloads[1]["history"]],
                         ["One bottle in ORD-2026-00031 arrived dented.", "Reply 1"])
        texts = [element.value for element in app.markdown]
        self.assertIn("Reply 2", texts)
        for heading in ("#### Supervisor · decide", "#### Specialist · returns refunds (react)",
                        "#### Approver · proposal check", "#### Gateway · write", "#### Approver · reply check"):
            self.assertIn(heading, texts)
        successes = [element.value for element in app.success]
        self.assertTrue(any(value.startswith("Recorded refund 32") for value in successes))
        self.assertIn("Resolved", successes)

    def test_unreachable_api_shows_an_error_instead_of_crashing(self):
        from streamlit.testing.v1 import AppTest

        with patch.dict("os.environ", {"SUPPORT_OPS_MULTI_API": "http://127.0.0.1:9"}):
            app = AppTest.from_file(str(ROOT / "scripts/demo_multi_agent.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertIn("cannot reach the multi-agent API", app.sidebar.error[0].value)


if __name__ == "__main__":
    unittest.main()
