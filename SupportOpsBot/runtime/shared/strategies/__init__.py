"""The two strategies, one loop each for every agent: the single agent, the supervisor and the specialists.

react         one step at a time, each chosen after seeing the previous result.
plan_execute  plan all steps up front, run them, then replan only on an error or for a missing step.

An agent is its prompt and its toolset; what its tools do (read, write, ask a specialist) is the setup's
business. A toolset with a terminal tool (finish) ends when the agent calls it; one without ends when
the agent stops calling tools. Either way the loop also ends when a gate stops the turn, and every step
draws on the ticket's budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from SupportOpsBot.runtime.shared.context import TurnContext
from SupportOpsBot.runtime.shared.tools import Toolset

STRATEGIES = ("react", "plan_execute")


@dataclass(frozen=True)
class Agent:
    name: str                  # in the trace and the evidence: agent, supervisor, orders_shipping, ...
    role: str                  # the LLM role: agent, supervisor or specialist
    prompt: str                # the system prompt, before the strategy's own instructions
    planner_context: str = ""  # retrieval overview given up front, e.g. available policy files or sections


@dataclass
class AgentRun:
    finished: dict[str, Any] | None = None   # the terminal tool's arguments, when the agent called it
    summary: str | None = None               # the agent's last words, when it stopped without a terminal tool
    error: str | None = None                 # why the loop gave up, e.g. the same error three times
    calls: list[dict[str, Any]] = field(default_factory=list)   # this run's tool calls and results


def run_agent(strategy: str, ctx: TurnContext, agent: Agent, tools: Toolset, brief: dict[str, Any]) -> AgentRun:
    from SupportOpsBot.runtime.shared.strategies.plan_execute import plan_execute
    from SupportOpsBot.runtime.shared.strategies.react import react

    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}, not {strategy!r}")
    ctx.log("agent_start", agent=agent.name, strategy=strategy)
    run = (react if strategy == "react" else plan_execute)(ctx, agent, tools, brief)
    ctx.log("agent_end", agent=agent.name, finished=run.finished, error=run.error, stopped=ctx.stop_reason)
    return run
