"""The one react loop and the one plan_execute loop, on a small toolset (every agent uses these two)."""
import pytest

from support_ops.shared.budget import BudgetExceeded
from support_ops.shared.strategies import Agent, run_agent
from support_ops.shared.tools import OrderArguments, Tool, Toolset, finish_tool
from tests.fakes import ScriptedLLM, as_json, plan, services, text, tool_call

AGENT = Agent("agent", "agent", "You help.")


def toolset(results=None, *, finish=True, data_runs=None):
    """`lookup` returns the next of `results` (a dict, or an exception to raise)."""
    results = list(results or [])
    runs = data_runs if data_runs is not None else []

    def lookup(args):
        runs.append(args)
        value = results.pop(0) if results else {"found": args["order_id"]}
        if isinstance(value, Exception):
            raise value
        return value

    tools = [Tool("lookup", OrderArguments, "Look an order up.", lookup)]
    return Toolset(tools + ([finish_tool()] if finish else []), terminal="finish" if finish else None)


def run(strategy, llm, tools, **service_args):
    ctx = services(llm=llm, **service_args).ctx
    return run_agent(strategy, ctx, AGENT, tools, {"message": "Where is O1?"}), ctx


# ----------------------------------------------------------------------------- react


def test_react_runs_one_call_per_turn_and_answers_only_that_call():
    llm = ScriptedLLM().push("agent", tool_call("lookup", order_id="O1"),
                             tool_call("finish", outcome="resolved", note="Found."))
    llm.queues["agent"][0].tool_calls.append(tool_call("lookup", "second", order_id="O2").tool_calls[0])
    result, ctx = run("react", llm, toolset())
    assert result.finished == {"outcome": "resolved", "note": "Found."}
    assert [item["arguments"] for item in ctx.evidence] == [{"order_id": "O1"}]   # the second call was deferred
    assert ctx.tracer.of_kind("tool_calls_deferred")[0]["tools"] == ["lookup"]
    second = [messages for role, messages, _ in llm.calls if role == "agent"][1]
    assert [message["role"] for message in second[-2:]] == ["assistant", "tool"]
    assert len(second[-2]["tool_calls"]) == 1


def test_react_gives_errors_back_and_stops_on_the_same_error_three_times():
    llm = ScriptedLLM().push("agent", *[tool_call("lookup", order_id="")] * 3)
    result, ctx = run("react", llm, toolset())
    assert result.error.startswith("lookup failed 3 times in a row")
    assert all("error" in item["result"] for item in ctx.evidence)


def test_react_without_a_terminal_tool_ends_when_the_agent_stops_calling_tools():
    llm = ScriptedLLM().push("agent", tool_call("lookup", order_id="O1"), text("I have what I need."))
    result, _ = run("react", llm, toolset(finish=False))
    assert (result.finished, result.summary) == (None, "I have what I need.")
    assert "reply with a one-line summary and no tool call" in llm.calls[0][1][0]["content"]


def test_a_failure_the_agent_cannot_fix_is_retried_once():
    runs = []
    llm = ScriptedLLM().push("agent", tool_call("lookup", order_id="O1"),
                             tool_call("finish", outcome="resolved", note="ok"))
    _, ctx = run("react", llm, toolset([ConnectionError("database timeout"), {"found": "O1"}], data_runs=runs))
    assert len(runs) == 2 and ctx.evidence[0]["result"] == {"found": "O1"}
    assert ctx.tracer.of_kind("tool_retry")[0]["error"] == "database timeout"
    assert ctx.budget.tool_calls == 1   # a retry is the same tool call


def test_every_agent_step_and_tool_call_draws_on_the_ticket_budget():
    llm = ScriptedLLM().push("agent", *[tool_call("lookup", order_id="O1")] * 20)
    with pytest.raises(BudgetExceeded, match="agent step limit"):
        run("react", llm, toolset())
    llm = ScriptedLLM().push("agent", plan(*[("lookup", {"order_id": "O1"})] * 41))
    result, ctx = run("plan_execute", llm.push("agent", plan(("finish", {"outcome": "resolved", "note": "x"}))),
                      toolset())
    assert ctx.tracer.of_kind("plan_rejected")[0]["reason"] == "invalid plan: the plan exceeds the remaining tool budget"


# ----------------------------------------------------------------------------- plan_execute


def test_plan_execute_runs_the_plan_then_checks_for_a_missing_step():
    llm = ScriptedLLM().push("agent", plan(("lookup", {"order_id": "O1"})),
                             plan(("lookup", {"order_id": "O2"}), ("finish", {"outcome": "resolved", "note": "Both."})))
    result, ctx = run("plan_execute", llm, toolset())
    assert result.finished["note"] == "Both."
    assert [item["arguments"]["order_id"] for item in ctx.evidence] == ["O1", "O2"]
    check = llm.user_payload("agent", 1)
    assert check["evidence"][0]["result"] == {"found": "O1"}   # the completion check sees the results
    assert "Plan only the calls still needed" in llm.calls[1][1][0]["content"]
    assert "Plan only the calls still needed" not in llm.calls[0][1][0]["content"]


def test_a_failed_call_stops_the_plan_and_the_correction_sees_why():
    llm = ScriptedLLM().push("agent", plan(("lookup", {"order_id": "O1"}), ("lookup", {"order_id": "O2"})),
                             plan(("finish", {"outcome": "awaiting_customer", "note": "Which order?"})))
    runs = []
    result, ctx = run("plan_execute", llm, toolset([{"error": "order O1 not found"}], data_runs=runs))
    assert len(runs) == 1   # O2 never ran on a plan that had already failed
    assert "order O1 not found" in llm.user_payload("agent", 1)["previous_error"]
    assert result.finished["outcome"] == "awaiting_customer"


def test_blocked_and_review_required_results_also_replan():
    llm = ScriptedLLM().push("agent", plan(("lookup", {"order_id": "O1"}), ("lookup", {"order_id": "O2"})),
                             plan(("finish", {"outcome": "resolved", "note": "ok"})))
    runs = []
    run("plan_execute", llm, toolset([{"status": "review_required", "reason": "review first"}], data_runs=runs))
    assert len(runs) == 1 and "review_required: review first" in llm.user_payload("agent", 1)["previous_error"]


def test_invalid_plans_are_corrected_up_to_max_replans():
    bad = [as_json({"steps": []}), plan(("finish", {"outcome": "resolved", "note": "x"}),
                                        ("lookup", {"order_id": "O1"})), plan(), text("not json")]
    result, ctx = run("plan_execute", ScriptedLLM().push("agent", *bad), toolset())
    reasons = [event["reason"] for event in ctx.tracer.of_kind("plan_rejected")]
    assert reasons[1] == "invalid plan: finish can only be the last call"
    assert reasons[2] == "invalid plan: nothing left to do: end with finish"
    assert result.error.startswith("no plan worked after 3 corrections")


def test_without_a_terminal_tool_an_empty_completion_check_ends_the_run():
    llm = ScriptedLLM().push("agent", plan(("lookup", {"order_id": "O1"})), plan())
    result, ctx = run("plan_execute", llm, toolset(finish=False))
    assert (result.finished, result.error, len(ctx.evidence)) == (None, None, 1)
    assert 'Return {"calls": []} when nothing is needed.' in llm.calls[0][1][0]["content"]


def test_the_planner_sees_the_agents_context_and_the_tool_schemas():
    agent = Agent("agent", "agent", "You help.", planner_context="Policy sections: Returns.")
    ctx = services(llm=ScriptedLLM().push("agent", plan(("finish", {"outcome": "resolved", "note": "x"})))).ctx
    run_agent("plan_execute", ctx, agent, toolset(), {"message": "hi"})
    system = ctx.llm.calls[0][1][0]["content"]
    assert "Policy sections: Returns." in system and '"name":"lookup"' in system
    assert '"title"' not in system   # pydantic's generated titles only cost tokens


def test_an_unknown_strategy_is_an_error():
    with pytest.raises(ValueError, match="react, plan_execute"):
        run("plan_and_execute", ScriptedLLM(), toolset())


def test_the_supervisor_plans_parts_and_a_worker_plans_its_calls_with_the_same_loop():
    for role, words, check in (("supervisor", "one ask_specialist call per part", "Plan only the parts still open"),
                               ("agent", "Plan the tool calls you need for this case", "Plan only the calls still needed")):
        llm = ScriptedLLM().push(role, plan(("lookup", {"order_id": "O1"})),
                                 plan(("finish", {"outcome": "resolved", "note": "x"})))
        ctx = services(llm=llm).ctx
        run_agent("plan_execute", ctx, Agent("a", role, "You help."), toolset(), {"message": "hi"})
        first, second = (messages[0]["content"] for _, messages, _ in llm.calls)
        assert words in first and check in second and check not in first
