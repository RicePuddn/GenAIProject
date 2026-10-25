"""Tests for the shared helpers: clock, config, tracer and the scripted LLM."""

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
