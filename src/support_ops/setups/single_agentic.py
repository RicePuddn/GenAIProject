"""S1 (RAG) and S2 (skills): one agent holds every tool, backend_write and request_review included, and
decides for itself what to read, when to write, when to ask for review and when it is finished."""
from __future__ import annotations

from support_ops import prompts
from support_ops.setups.base import Setup, Turn, TurnResult, finished
from support_ops.shared.strategies import Agent, run_agent
from support_ops.shared.tools import ALL_TOOLS, finish_tool


class SingleAgentic(Setup):
    agents, control = "single", "agentic"

    def run(self, turn: Turn) -> TurnResult:
        agent = Agent("agent", "agent", prompts.load("agent", today=turn.today, facts=turn.facts(),
                                                     control=prompts.load("control", "agentic")),
                      turn.policy.planner_context())
        tools = turn.registry.toolset(ALL_TOOLS, extra=[finish_tool()], terminal="finish")
        return finished(turn, run_agent(self.strategy["agent"], turn.ctx, agent, tools, turn.brief))
