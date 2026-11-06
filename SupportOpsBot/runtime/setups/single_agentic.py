"""S1 (RAG) and S2 (skills): one agent holds every registered tool, including named writes and request_review, and
decides for itself what to read, when to write, when to ask for review and when it is finished."""
from __future__ import annotations

from SupportOpsBot.runtime import prompts
from SupportOpsBot.runtime.setups.base import Setup, Turn, TurnResult, finished
from SupportOpsBot.runtime.shared.strategies import Agent, run_agent
from SupportOpsBot.runtime.shared.tools import finish_tool
from SupportOpsBot.runtime.shared.tool_catalog import ALL_TOOLS


class SingleAgentic(Setup):
    agents, control = "single", "agentic"

    def run(self, turn: Turn) -> TurnResult:
        agent = Agent("agent", "agent", prompts.load("agent", today=turn.today, facts=turn.facts(),
                                                     control=prompts.load("control", "agentic")),
                      turn.policy.planner_context())
        tools = turn.registry.toolset(ALL_TOOLS, extra=[finish_tool()], terminal="finish")
        return finished(turn, run_agent(self.strategy["agent"], turn.ctx, agent, tools, turn.brief))
