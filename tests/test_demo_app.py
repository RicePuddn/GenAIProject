"""The Streamlit demo page, driven by Streamlit's AppTest against a fake API (no server needed)."""
import importlib.util
import io
import json
import unittest
from unittest.mock import patch

from support_ops.helpers.config import ROOT

INFO = {"policy_mode": "rag", "corpus": "mini", "default_strategy": "react",
        "model": "openai/gpt-oss-120b", "reasoning_effort": "medium"}
CUSTOMERS = [{"id": "CUS-0001", "first_name": "Ana", "last_name": "Lim", "email": "ana@example.test",
              "loyalty_tier": "gold"}]


def chat_response(message, history, reply):
    clauses = [{"id": "RET-1", "title": "Return window", "score": 3.1, "via": "search", "text": "30 days."}]
    execution = {"kind": "tool_end", "call_id": "call-1", "tool": "policy", "elapsed_s": 0.2,
                 "output": {"query": "return window", "clauses": clauses}}
    requested = {"kind": "tool_call_requested", "call_id": "call-1", "tool": "policy",
                 "input": {"action": "search", "query": "return window"}}
    return {
        "ticket_id": "TKT-DEMO", "strategy": "react", "reply": reply, "ticket_status": "open",
        "handoff_packet": None, "evidence": [], "tool_call_count": 1,
        "tool_calls": [{**requested, "execution": execution}],
        "trace": {"llm_calls": 3, "input_tokens": 100, "output_tokens": 20, "cost_usd": 0.0004,
                  "elapsed_s": 1.5, "tool_call_count": 1, "events": [
                      {"kind": "guard_classification", "label": "benign", "malicious_score": 0.001},
                      {"kind": "llm_response", "role": "supervisor", "tool_calls": ["policy"],
                       "reasoning": "I need the return window."},
                      requested, execution]},
        "history": [*history, {"role": "customer", "content": message}, {"role": "agent", "content": reply}],
    }


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class FakeApi:
    """/info as JSON; /chat/stream as Server-Sent Events: each trace event, then the result."""

    def __init__(self):
        self.chat_payloads = []

    def __call__(self, request, timeout=None):
        if request.full_url.endswith("/info"):
            return io.BytesIO(json.dumps(INFO).encode())
        if request.full_url.endswith("/customers"):
            return io.BytesIO(json.dumps(CUSTOMERS).encode())
        self.assert_stream(request)
        payload = json.loads(request.data)
        self.chat_payloads.append(payload)
        response = chat_response(payload["customer_message"], payload["history"], f"Reply {len(self.chat_payloads)}")
        body = "".join(sse("trace", event) for event in response["trace"]["events"]) + sse("result", response)
        return io.BytesIO(body.encode())

    @staticmethod
    def assert_stream(request):
        assert request.full_url.endswith("/chat/stream"), request.full_url


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "demo extra not installed")
class DemoAppTests(unittest.TestCase):
    def test_chat_shows_steps_and_sends_history_on_the_next_turn(self):
        from streamlit.testing.v1 import AppTest

        api = FakeApi()
        with patch("urllib.request.urlopen", api):
            app = AppTest.from_file(str(ROOT / "scripts/demo_app.py"), default_timeout=30).run()
            self.assertIn("rag", app.sidebar.markdown[0].value)
            app.chat_input[0].set_value("Can I return my headphones?").run()
            app.chat_input[0].set_value("It's ORD-2026-00041.").run()

        self.assertFalse(app.exception)
        self.assertEqual(api.chat_payloads[0]["history"], [])
        self.assertEqual(api.chat_payloads[1]["ticket_id"], "TKT-DEMO")
        self.assertEqual([turn["content"] for turn in api.chat_payloads[1]["history"]],
                         ["Can I return my headphones?", "Reply 1"])
        texts = [element.value for element in app.markdown]
        self.assertIn("Reply 2", texts)
        self.assertIn("I need the return window.", [element.value for element in app.info])
        self.assertTrue(any("`policy`" in text for text in texts))
        self.assertEqual(len(app.dataframe), 2)   # retrieved clauses, once per exchange

    def test_stream_errors_are_shown_and_nothing_is_added_to_the_history(self):
        from streamlit.testing.v1 import AppTest

        def failing(request, timeout=None):
            if request.full_url.endswith("/info"):
                return io.BytesIO(json.dumps(INFO).encode())
            if request.full_url.endswith("/customers"):
                return io.BytesIO(json.dumps(CUSTOMERS).encode())
            return io.BytesIO(sse("error", {"detail": "set GROQ_API_KEY"}).encode())

        with patch("urllib.request.urlopen", failing):
            app = AppTest.from_file(str(ROOT / "scripts/demo_app.py"), default_timeout=30).run()
            app.chat_input[0].set_value("Hello").run()
        self.assertIn("set GROQ_API_KEY", [element.value for element in app.error])
        self.assertEqual(app.session_state.history, [])

    def test_new_ticket_clears_the_conversation(self):
        from streamlit.testing.v1 import AppTest

        with patch("urllib.request.urlopen", FakeApi()):
            app = AppTest.from_file(str(ROOT / "scripts/demo_app.py"), default_timeout=30).run()
            app.chat_input[0].set_value("Hello").run()
            app.sidebar.button[0].click().run()
        self.assertEqual(app.session_state.history, [])
        self.assertIsNone(app.session_state.ticket_id)

    def test_the_signed_in_account_is_sent_and_switching_it_starts_a_new_ticket(self):
        from streamlit.testing.v1 import AppTest

        api = FakeApi()
        with patch("urllib.request.urlopen", api):
            app = AppTest.from_file(str(ROOT / "scripts/demo_app.py"), default_timeout=30).run()
            app.chat_input[0].set_value("Hello").run()
            self.assertNotIn("customer_id", api.chat_payloads[0])   # signed out: nothing is scoped
            app.sidebar.selectbox[0].set_value("CUS-0001").run()
            self.assertEqual((app.session_state.history, app.session_state.ticket_id), ([], None))
            app.chat_input[0].set_value("Where is my order?").run()
        self.assertFalse(app.exception)
        self.assertEqual(api.chat_payloads[1]["customer_id"], "CUS-0001")
        self.assertEqual(api.chat_payloads[1]["history"], [])


if __name__ == "__main__":
    unittest.main()
