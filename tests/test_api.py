"""The one API: any setup by request or config, run.clock_start unless the request overrides it."""
from functools import partial
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from support_ops import api
from support_ops.shared.pipeline import run_turn
from tests.fakes import FakeDatabase, ScriptedLLM, tool_call

client = TestClient(api.app)


def chat(body, llm):
    db = FakeDatabase()
    with patch("support_ops.api._llm", return_value=llm), patch("support_ops.api.signed_in", lambda account: account), \
            patch("support_ops.api.run_turn", Mock(wraps=partial(run_turn, reader=db, writer=db))) as turn:
        response = client.post("/chat", json=body)
    return response, turn


def finished():
    return ScriptedLLM().push("agent", tool_call("finish", outcome="resolved", note="Hello."))


def test_info_lists_every_setup_without_a_key_or_database():
    info = client.get("/info").json()
    assert info["default_setup"] == "s1-a" and len(info["setups"]) == 24
    assert info["setups"]["s6-b"] == {"name": "S6", "variant": "b", "agents": "single", "control": "graph",
                                      "policy_mode": "skills", "strategy": {"agent": "plan_execute"}}
    assert info["clock_start"] == "2026-10-01T09:00:00+08:00"


def test_a_request_names_its_setup_and_gets_the_same_response_shape():
    response, _ = chat({"customer_message": "Hi", "setup": "s2-a", "ticket_id": "TKT-1"}, finished())
    body = response.json()
    assert response.status_code == 200, body
    assert (body["setup"], body["strategy"], body["ticket_status"]) == ("s2-a", {"agent": "react"}, "resolved")
    assert body["history"] == [{"role": "customer", "content": "Hi"}, {"role": "agent", "content": body["reply"]}]


def test_the_workflow_date_is_run_clock_start_unless_the_request_overrides_it():
    for body, expected in (({}, "2026-10-01T09:00:00+08:00"),
                           ({"clock_start": "2026-11-05T10:00:00+00:00"}, "2026-11-05T10:00:00+00:00")):
        response, turn = chat({"customer_message": "Hi", "setup": "s2-a", **body}, finished())
        assert response.status_code == 200
        assert turn.call_args.kwargs["clock"].now().isoformat() == expected


def test_bad_setups_and_naive_dates_are_rejected_before_any_turn_runs():
    for body in ({"setup": "s9-a"}, {"clock_start": "2026-11-05T10:00:00"}):
        response, turn = chat({"customer_message": "Hi", **body}, finished())
        assert response.status_code == 422
        turn.assert_not_called()
    response, turn = chat({"customer_message": "Hi", "setup": "s1-c"}, finished())
    assert (response.status_code, response.json()["detail"]) == (503, "S1 has variants a, b, not c")
    turn.assert_not_called()
