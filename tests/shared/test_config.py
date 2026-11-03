"""Setup IDs resolve from configs/setups.yaml, and every value held constant comes from base.yaml."""
from unittest.mock import Mock

import pytest

from support_ops.setups import build_setup
from support_ops.setups.multi_agentic import MultiAgentic
from support_ops.setups.multi_graph import MultiGraph
from support_ops.setups.single_agentic import SingleAgentic
from support_ops.setups.single_graph import SingleGraph
from support_ops.shared.config import load_config, resolve_setup, setup_ids
from support_ops.shared.llm import OpenAICompatLLM


def test_there_are_24_variants_two_per_single_setup_and_four_per_multi_setup():
    ids = setup_ids()
    assert len(ids) == 24
    assert [i for i in ids if i.startswith("s1-")] == ["s1-a", "s1-b"]
    assert [i for i in ids if i.startswith("s7-")] == ["s7-a", "s7-b", "s7-c", "s7-d"]


@pytest.mark.parametrize("setup_id, expected", [
    ("s1-a", {"setup": {"name": "S1", "agents": "single", "control": "agentic"}, "mode": "rag",
              "strategy": {"agent": "react"}}),
    ("s6-b", {"setup": {"name": "S6", "agents": "single", "control": "graph"}, "mode": "skills",
              "strategy": {"agent": "plan_execute"}}),
    ("s3-c", {"setup": {"name": "S3", "agents": "multi", "control": "agentic"}, "mode": "rag",
              "strategy": {"supervisor": "react", "specialists": "plan_execute"}}),
    ("s8-d", {"setup": {"name": "S8", "agents": "multi", "control": "graph"}, "mode": "skills",
              "strategy": {"supervisor": "plan_execute", "specialists": "react"}}),
])
def test_a_setup_id_resolves_its_agents_control_policy_and_strategies(setup_id, expected):
    cfg = load_config(setup=setup_id)
    assert {key: cfg["setup"][key] for key in expected["setup"]} == expected["setup"]
    assert (cfg["setup"]["id"], cfg["run"]["setup"]) == (setup_id, setup_id)
    assert cfg["policy"]["mode"] == expected["mode"]
    assert cfg["strategy"] == expected["strategy"]


def test_each_pair_runs_the_same_code_with_only_the_policy_mode_changed():
    classes = {"S1": SingleAgentic, "S3": MultiAgentic, "S5": SingleGraph, "S7": MultiGraph}
    for first, cls in classes.items():
        second = f"S{int(first[1]) + 1}"
        for variant in "ab":
            a, b = load_config(setup=f"{first.lower()}-{variant}"), load_config(setup=f"{second.lower()}-{variant}")
            assert type(build_setup(cfg=a)) is type(build_setup(cfg=b)) is cls
            differences = {key for key in a if a[key] != b[key]}
            assert differences <= {"setup", "policy", "run"}
            assert {k for k in a["policy"] if a["policy"][k] != b["policy"][k]} == {"mode"}


def test_unknown_setups_and_variants_are_errors():
    for bad in ("s9-a", "s1", "S1-A-x", "s1-c"):
        with pytest.raises(ValueError):
            resolve_setup(bad)
    assert resolve_setup("S3-C")["setup"]["id"] == "s3-c"


def test_the_default_setup_and_overrides_come_after_the_setup():
    assert load_config()["setup"]["id"] == load_config()["run"]["setup"] == "s1-a"
    assert load_config(overrides={"run.setup": "s4-b"})["setup"]["id"] == "s4-b"
    assert load_config(setup="s4-b", overrides={"policy.mode": "rag"})["policy"]["mode"] == "rag"


def test_a_missing_key_is_an_error_not_a_different_setup():
    cfg = load_config()
    del cfg["llm"]["models"]
    with pytest.raises(KeyError):
        OpenAICompatLLM(cfg)
    cfg = load_config()
    del cfg["policy"]["mode"]
    from support_ops.shared.policy.provider import policy_from_config
    with pytest.raises(KeyError):
        policy_from_config(cfg, Mock())


def test_the_held_constant_values_are_the_experiment_s():
    cfg = load_config()
    assert set(cfg["llm"]["models"]) == {"agent", "supervisor", "specialist", "approver", "reply_writer", "guard"}
    assert set(cfg["llm"]["models"].values()) == {"lfm3:latest"}
    assert cfg["llm"]["provider"] == "ollama"
    assert cfg["llm"]["temperature"] == 0.0
    assert cfg["guard"]["threshold"] == 0.5
    assert (cfg["review"]["max_rejections"], cfg["review"]["max_reply_drafts"]) == (1, 2)
    assert (cfg["limits"]["max_tool_calls"], cfg["limits"]["max_agent_steps"], cfg["limits"]["max_cost_usd"]) == (40, 30, 0.05)
    assert cfg["run"]["clock_start"] == "2026-10-01T09:00:00+08:00"
    assert cfg["llm"]["cache_dir"] is None   # repeat trials must not replay the first


def test_the_trial_number_is_part_of_the_cache_key(tmp_path):
    cfg = load_config(overrides={"llm.cache_dir": str(tmp_path)})
    keys = []
    for trial in (0, 1):
        cfg["run"]["trial"] = trial
        llm = OpenAICompatLLM(cfg)
        response = Mock()
        response.choices = [Mock(message=Mock(content="hi", tool_calls=None, reasoning=None))]
        response.usage = Mock(prompt_tokens=1, completion_tokens=1)
        llm.client = Mock()
        llm.client.chat.completions.create.return_value = response
        llm.chat("agent", [{"role": "user", "content": "same request"}])
        keys.append({path.name for path in tmp_path.iterdir()})
    assert len(keys[1]) == 2   # the second trial made its own call instead of replaying the first
