"""react: one tool call per turn, each chosen after seeing the previous result."""
from __future__ import annotations

import json
from typing import Any

from SupportOpsBot.runtime import prompts
from SupportOpsBot.runtime.shared.context import TurnContext
from SupportOpsBot.runtime.shared.llm import ChatResult, ToolCall
from SupportOpsBot.runtime.shared.strategies import Agent, AgentRun
from SupportOpsBot.runtime.shared.tools import Toolset

# The same tool failing with the same error this many times in a row ends the run.
MAX_REPEATED_ERRORS = 3


def react(ctx: TurnContext, agent: Agent, tools: Toolset, brief: dict[str, Any]) -> AgentRun:
    end = prompts.load("strategies", "react_finish" if tools.terminal else "react_stop")
    system = f"{agent.prompt}\n\n{prompts.load('strategies', 'react')} {end}"
    if agent.planner_context:
        system += f"\n\nPolicy retrieval context:\n{agent.planner_context}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(brief, default=str)},
    ]
    run, nudged, last_error, repeats = AgentRun(), False, None, 0
    while not ctx.stopped:
        ctx.budget.agent_step()
        reply = ctx.chat(agent.role, messages, tools=tools.schemas)
        if not reply.tool_calls:
            if tools.terminal and not nudged:
                nudged = True
                messages += [{"role": "assistant", "content": reply.content or ""},
                             {"role": "user", "content": prompts.load("strategies", "nudge")}]
                continue
            run.summary = (reply.content or "").strip() or None
            return run
        chosen = reply.tool_calls[0]
        resolved_name, resolved_arguments = tools.resolve_terminal_call(chosen.name, chosen.arguments)
        if resolved_name != chosen.name:
            ctx.log("terminal_tool_name_repaired", agent=agent.name, generated=chosen.name,
                    resolved=resolved_name)
            chosen = ToolCall(chosen.id, resolved_name, resolved_arguments)
        if len(reply.tool_calls) > 1:
            ctx.log("tool_calls_deferred", agent=agent.name, tools=[call.name for call in reply.tool_calls[1:]])
        call_id = ctx.next_call_id()
        # Answer exactly the call that runs; the agent chooses again after seeing its result.
        messages.append(ChatResult(None, [ToolCall(call_id, chosen.name, chosen.arguments)]).as_message())
        value = tools.call(ctx, agent.name, call_id, chosen.name, chosen.arguments)
        run.calls.append({"tool": chosen.name, "arguments": chosen.arguments, "result": value})
        messages.append({"role": "tool", "tool_call_id": call_id, "name": chosen.name,
                         "content": json.dumps(value, default=str)})
        if chosen.name == tools.terminal and "error" not in value:
            run.finished = tools.validate(chosen.name, chosen.arguments)
            return run
        error = (chosen.name, value["error"]) if "error" in value else None
        repeats = repeats + 1 if error and error == last_error else (1 if error else 0)
        last_error = error
        if repeats >= MAX_REPEATED_ERRORS:
            ctx.log("repeated_error", agent=agent.name, tool=chosen.name, error=value["error"], times=repeats)
            run.error = f"{chosen.name} failed {repeats} times in a row with: {value['error']}"
            return run
    return run
