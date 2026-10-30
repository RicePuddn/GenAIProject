"""Tests for the shared helpers: clock, config, tracer and the scripted LLM."""

import json
import os
from datetime import datetime, timezone

import pytest

from support_ops.helpers.clock import Clock
from support_ops.helpers.config import ROOT, load_config, load_env
from support_ops.helpers.llm import ScriptedLLM, text, tool_call
from support_ops.helpers.tracing import Tracer


def test_clock_is_controllable_and_timezone_aware():
    start = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    clock = Clock(start)
    assert clock.now() == start
    assert (clock.advance(3600) - start).total_seconds() == 3600
    with pytest.raises(ValueError):
        Clock(datetime(2026, 10, 1))


def test_config_loads_base_and_applies_overrides():
    assert (ROOT / "configs" / "base.yaml").exists()
    cfg = load_config(overrides={"run.k_trials": 1, "llm.models.supervisor": "small"})
    assert cfg["run"]["k_trials"] == 1
    assert cfg["llm"]["models"]["supervisor"] == "small"
    assert cfg["llm"]["models"]["specialist"]          # untouched keys survive


def test_env_file_is_loaded_but_shell_variables_win(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("SUPPORT_OPS_TEST_KEY=from-file\nSUPPORT_OPS_TEST_OTHER=from-file\n")
    monkeypatch.delenv("SUPPORT_OPS_TEST_KEY", raising=False)
    monkeypatch.setenv("SUPPORT_OPS_TEST_OTHER", "from-shell")
    assert load_env(env)
    assert os.environ["SUPPORT_OPS_TEST_KEY"] == "from-file"
    assert os.environ["SUPPORT_OPS_TEST_OTHER"] == "from-shell"
    monkeypatch.delenv("SUPPORT_OPS_TEST_KEY")
    assert not load_env(tmp_path / "missing.env")


def test_scripted_llm_replays_in_order_and_records_usage():
    tracer = Tracer("T1")
    llm = ScriptedLLM().push("specialist", tool_call("get_order", order_id="O1"), text("done"))
    first = llm.chat("specialist", [{"role": "user", "content": "hi"}], tracer=tracer)
    assert first.tool_calls[0].name == "get_order"
    assert first.as_message()["tool_calls"][0]["function"]["arguments"] == '{"order_id": "O1"}'
    assert llm.chat("specialist", [], tracer=tracer).content == "done"
    assert tracer.llm_calls == 2 and tracer.input_tokens > 0
    with pytest.raises(AssertionError):
        llm.chat("specialist", [])                    # script exhausted


def test_tracer_dump(tmp_path):
    tracer = Tracer("T1")
    tracer.log("policy_read", source="skills", clause_ids=["RET-1"])
    tracer.dump(tmp_path / "t.json")
    assert '"RET-1"' in (tmp_path / "t.json").read_text()
    assert tracer.of_kind("policy_read")[0]["clause_ids"] == ["RET-1"]


def _groq_error(code: str, failed_generation: str):
    """A 400 shaped like Groq's: the openai client puts the inner `error` object in `body`."""
    import httpx
    from openai import BadRequestError

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"message": "Tool call validation failed", "type": "invalid_request_error",
            "code": code, "failed_generation": failed_generation}
    return BadRequestError("Error code: 400", response=httpx.Response(400, request=request), body=body)


def _llm_raising(error):
    from unittest.mock import Mock, patch

    from support_ops.helpers.llm import OpenAICompatLLM

    with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
        llm = OpenAICompatLLM(load_config(overrides={"llm.cache_dir": None}))
    llm.client = Mock()
    llm.client.chat.completions.create.side_effect = error
    return llm


def test_a_tool_call_groq_rejects_is_recovered_for_our_own_validation():
    # The call from a real S4 run: order_item_id is an alias our models accept, but not in the schema.
    attempt = {"name": "route", "arguments": {"action": "return", "order_id": "ORD-2026-00031",
                                              "items": [{"order_item_id": "ITM-0031-2", "quantity": 1}]}}
    llm = _llm_raising(_groq_error("tool_use_failed", json.dumps(attempt)))
    tracer = Tracer("T1")
    result = llm.chat("specialist", [{"role": "user", "content": "refund"}], tools=[{"type": "function"}], tracer=tracer)
    assert [(call.name, call.arguments) for call in result.tool_calls] == [("route", attempt["arguments"])]
    assert result.tool_calls[0].id.startswith("recovered_")
    assert tracer.of_kind("tool_call_recovered")[0]["tools"] == ["route"]


def test_other_provider_errors_still_raise():
    from openai import BadRequestError

    for error in (_groq_error("context_length_exceeded", ""), _groq_error("tool_use_failed", "not json")):
        llm = _llm_raising(error)
        with pytest.raises(BadRequestError):
            llm.chat("specialist", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    llm = _llm_raising(_groq_error("tool_use_failed", '{"name": "route", "arguments": {}}'))
    with pytest.raises(BadRequestError):   # no tools were offered, so there is nothing to recover
        llm.chat("supervisor", [{"role": "user", "content": "hi"}])
