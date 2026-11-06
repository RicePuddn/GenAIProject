"""plan_execute: plan every call up front and run the plan; replan only on an error or for a missing step.

One loop on two levels: a supervisor plans the parts of the message (its calls are ask_specialist, one
per part), a working agent plans its own tool calls. Only the planning text differs.

A reply that is not a plan at all (not JSON, or not {"calls": [...]}) is asked for again, as every JSON reply
is, `retries.invalid_json` times; then the run gives up. A correction (after a plan that names an unknown
tool, bad arguments or too many calls, or after a failed call) is a replan, up to `limits.max_replans`. After a
plan has run in full, the agent is asked once more, with the results, for anything still missing (a
completion check, up to `limits.max_plan_checks`): that is where a plan that needed a value it had not
seen yet goes on, and where an agent with a terminal tool finishes.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from SupportOpsBot.runtime import prompts
from SupportOpsBot.runtime.shared.context import TurnContext, chat_json
from SupportOpsBot.runtime.shared.strategies import Agent, AgentRun
from SupportOpsBot.runtime.shared.tools import Toolset, _argument_errors

# Results that change what comes next, so no later planned call runs on them.
REPLAN_STATUSES = {"blocked", "review_required"}


class PlannedCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    arguments: dict[str, Any]


class Plan(BaseModel):
    """The shape of a plan. Extra top-level keys (a model's own reasoning) are ignored."""
    calls: list[PlannedCall]


def plan_execute(ctx: TurnContext, agent: Agent, tools: Toolset, brief: dict[str, Any]) -> AgentRun:
    limits = ctx.cfg["limits"]
    run = AgentRun()
    corrections = checks = 0
    previous_error: str | None = None
    checking = False
    end = prompts.load("strategies", "plan_finish" if tools.terminal else "plan_stop")
    level = "parts" if agent.role == "supervisor" else "calls"
    plan_prompt = prompts.load("strategies", "plan", level=prompts.load("strategies", f"plan_{level}"),
                               tools=json.dumps(tools.schemas, separators=(",", ":")),
                               context=agent.planner_context, end=end)
    while not ctx.stopped:
        system = f"{agent.prompt}\n\n{plan_prompt}"
        if checking:
            system += "\n" + prompts.load("strategies", f"plan_check_{level}")
        try:
            plan = chat_json(ctx, agent.role, [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({**brief, "evidence": [*brief.get("evidence", []), *run.calls],
                                                        "previous_error": previous_error}, default=str)},
            ], Plan, before_call=ctx.budget.agent_step)
        except ValueError as exc:
            run.error = f"no readable plan: {exc}"
            return run
        try:
            calls = _validated(ctx, tools, plan, agent.name)
        except ValueError as exc:
            failure = f"invalid plan: {exc}"
        else:
            ctx.log("plan", agent=agent.name, calls=[[name, args] for name, args in calls], check=checking)
            if not calls:
                return run
            failure = None
            for name, args in calls:
                value = tools.call(ctx, agent.name, ctx.next_call_id(), name, args)
                run.calls.append({"tool": name, "arguments": args, "result": value})
                if ctx.stopped:
                    return run
                if name == tools.terminal and "error" not in value:
                    run.finished = args
                    return run
                if "error" in value or value.get("status") in REPLAN_STATUSES:
                    failure = (f"planned call {name} returned "
                               + (f"an error: {value['error']}" if "error" in value
                                  else f"{value['status']}: {value.get('reason')}"))
                    break
            if failure is None:   # the plan ran in full
                if checks >= limits["max_plan_checks"]:
                    run.error = f"still not finished after {checks} completion checks"
                    return run
                checks, checking, previous_error = checks + 1, True, None
                ctx.log("plan_check", agent=agent.name, check=checks)
                continue
        corrections += 1
        ctx.log("plan_rejected", agent=agent.name, reason=failure, replan=corrections)
        if corrections > limits["max_replans"]:
            run.error = f"no plan worked after {corrections} corrections: {failure}"
            return run
        previous_error, checking = failure, False
    return run


def _validated(ctx: TurnContext, tools: Toolset, plan: Plan, agent_name: str) -> list[tuple[str, dict[str, Any]]]:
    calls = []
    for call in plan.calls:
        name, arguments = tools.resolve_terminal_call(call.name, call.arguments)
        if name != call.name:
            ctx.log("terminal_tool_name_repaired", generated=call.name, resolved=name)
        try:
            validated = tools.validate(name, arguments)
        except ValidationError as exc:
            failures = ctx.invalid_argument_calls.get(agent_name, 0) + 1
            ctx.invalid_argument_calls[agent_name] = failures
            fields = _argument_errors(exc)
            if failures > 1:
                ctx.stop(f"{agent_name} made a second invalid tool call")
                raise ValueError("a second invalid tool call ended this agent run: " + "; ".join(fields)) from exc
            raise ValueError("invalid tool arguments; correct these fields and call once more: "
                             + "; ".join(fields)) from exc
        calls.append((name, validated))
    if sum(tools.tools[name].data for name, _ in calls) > ctx.budget.tool_calls_left:
        raise ValueError("the plan exceeds the remaining tool budget")
    if tools.terminal:
        names = [name for name, _ in calls]
        if not calls:
            raise ValueError(f"nothing left to do: end with {tools.terminal}")
        if tools.terminal in names[:-1]:
            raise ValueError(f"{tools.terminal} can only be the last call")
    return calls
