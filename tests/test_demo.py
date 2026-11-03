"""The Streamlit demo page, driven by Streamlit's AppTest. The page is served the real /info and the real
/chat/stream output of the API for a scripted turn, so the page and the API cannot drift apart."""
import importlib.util
import io
import json
import unittest
from functools import partial
from unittest.mock import patch

from support_ops import api
from support_ops.shared.config import ROOT, load_config
from support_ops.shared.pipeline import run_turn
from support_ops.shared.policy.rag import RagPolicy
from support_ops.shared.rules import Rules
from support_ops.shared.tracing import Tracer
from tests.fakes import FakeDatabase, FakeRetriever, text
from tests.setups.scripts import Scenario, policy_read, script

CUSTOMERS = [{"id": "CUS-0001", "first_name": "Ana", "last_name": "Lim", "email": "ana@example.test",
              "loyalty_tier": "gold"}]
DEMO = str(ROOT / "apps/demo.py")


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


class FakeApi:
    """urlopen for the page: /info and /customers as JSON, /chat/stream as the API's own events."""

    def __init__(self):
        self.chat_payloads = []
        self.db = FakeDatabase()

    def __call__(self, request, timeout=None):
        if request.full_url.endswith("/info"):
            return io.BytesIO(json.dumps(api.info()).encode())
        if request.full_url.endswith("/customers"):
            return io.BytesIO(json.dumps(CUSTOMERS).encode())
        assert request.full_url.endswith("/chat/stream"), request.full_url
        payload = json.loads(request.data)
        self.chat_payloads.append(payload)
        setup = payload.get("setup") or load_config()["setup"]["id"]
        mode = load_config(setup=setup)["policy"]["mode"]
        llm = script(setup, Scenario(reads=[policy_read(mode)], note=f"Reply {len(self.chat_payloads)}"))
        first = next(role for role in ("agent", "supervisor") if llm.queues[role])
        llm.queues[first][0].reasoning = "I need the return window."
        reply = text(f"Reply {len(self.chat_payloads)}")
        llm.on("reply_writer", lambda *_: reply)
        policy = RagPolicy(FakeRetriever(), Rules.load()) if mode == "rag" else None
        events = []
        tracer = Tracer(payload.get("ticket_id") or "TKT-DEMO", listener=events.append)
        with patch("support_ops.api._llm", return_value=llm), patch("support_ops.api.signed_in", lambda account: account), \
                patch("support_ops.api.run_turn", partial(run_turn, reader=self.db, writer=self.db, policy=policy)):
            response = api._run_chat(api.ChatRequest(**payload), tracer)
        body = "".join(sse("trace", event) for event in events) + sse("result", response.model_dump(mode="json"))
        return io.BytesIO(body.encode())


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "demo extra not installed")
class DemoTests(unittest.TestCase):
    @staticmethod
    def app():
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(DEMO, default_timeout=30)

    def test_chat_shows_steps_and_sends_history_on_the_next_turn(self):
        fake = FakeApi()
        with patch("urllib.request.urlopen", fake):
            app = self.app().run()
            self.assertIn("S1-a", app.sidebar.selectbox[0].format_func(app.sidebar.selectbox[0].value))
            app.chat_input[0].set_value("Can I return my headphones?").run()
            app.chat_input[0].set_value("It's ORD-2026-00001.").run()
        self.assertFalse(app.exception)
        self.assertEqual(fake.chat_payloads[0]["history"], [])
        self.assertEqual(fake.chat_payloads[0]["setup"], "s1-a")
        self.assertNotIn("clock_start", fake.chat_payloads[0])   # run.clock_start unless today is chosen
        self.assertEqual(fake.chat_payloads[1]["ticket_id"], "TKT-DEMO")
        self.assertEqual([turn["content"] for turn in fake.chat_payloads[1]["history"]],
                         ["Can I return my headphones?", "Reply 1"])
        texts = [element.value for element in app.markdown]
        self.assertIn("Reply 2", texts)
        self.assertIn("I need the return window.", [element.value for element in app.info])
        self.assertTrue(any("`policy`" in text for text in texts))
        self.assertEqual(len(app.dataframe), 2)   # retrieved clauses, once per exchange

    def test_any_setup_can_be_chosen_and_switching_starts_a_new_ticket(self):
        fake = FakeApi()
        with patch("urllib.request.urlopen", fake):
            app = self.app().run()
            self.assertEqual(len(app.sidebar.selectbox[0].options), 24)
            app.chat_input[0].set_value("Hello").run()
            app.sidebar.selectbox[0].set_value("s7-c").run()
            self.assertEqual((app.session_state.history, app.session_state.ticket_id), ([], None))
            app.chat_input[0].set_value("Where is my order?").run()
        self.assertFalse(app.exception)
        self.assertEqual(fake.chat_payloads[1]["setup"], "s7-c")
        self.assertTrue(any("Supervisor asks" in element.value for element in app.markdown))

    def test_today_s_date_is_an_explicit_choice(self):
        fake = FakeApi()
        with patch("urllib.request.urlopen", fake):
            app = self.app().run()
            app.sidebar.toggle[0].set_value(True).run()
            app.chat_input[0].set_value("Hello").run()
        self.assertIn("clock_start", fake.chat_payloads[0])

    def test_stream_errors_are_shown_and_nothing_is_added_to_the_history(self):
        def failing(request, timeout=None):
            if request.full_url.endswith("/info"):
                return io.BytesIO(json.dumps(api.info()).encode())
            if request.full_url.endswith("/customers"):
                return io.BytesIO(json.dumps(CUSTOMERS).encode())
            return io.BytesIO(sse("error", {"detail": "Ollama is unavailable"}).encode())

        with patch("urllib.request.urlopen", failing):
            app = self.app().run()
            app.chat_input[0].set_value("Hello").run()
        self.assertIn("Ollama is unavailable", [element.value for element in app.error])
        self.assertEqual(app.session_state.history, [])

    def test_new_ticket_clears_the_conversation(self):
        fake = FakeApi()
        with patch("urllib.request.urlopen", fake):
            app = self.app().run()
            app.chat_input[0].set_value("Hello").run()
            app.sidebar.button[0].click().run()
        self.assertEqual(app.session_state.history, [])
        self.assertIsNone(app.session_state.ticket_id)

    def test_the_signed_in_account_is_sent_and_switching_it_starts_a_new_ticket(self):
        fake = FakeApi()
        with patch("urllib.request.urlopen", fake):
            app = self.app().run()
            app.chat_input[0].set_value("Hello").run()
            self.assertNotIn("customer_id", fake.chat_payloads[0])   # signed out: nothing is scoped
            app.sidebar.selectbox[1].set_value("CUS-0001").run()
            self.assertEqual((app.session_state.history, app.session_state.ticket_id), ([], None))
            app.chat_input[0].set_value("Where is my order?").run()
        self.assertFalse(app.exception)
        self.assertEqual(fake.chat_payloads[1]["customer_id"], "CUS-0001")
        self.assertEqual(fake.chat_payloads[1]["history"], [])


if __name__ == "__main__":
    unittest.main()
