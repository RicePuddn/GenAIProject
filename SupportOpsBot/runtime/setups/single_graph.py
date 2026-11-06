"""S5 (RAG) and S6 (skills): one agent inside a fixed LangGraph sequence decided in code.

The agent gathers facts with read and policy tools, then proposes; the graph checks the rules, has any
write reviewed when required, and writes (shared/flow.py). The agent cannot write, start a review or end
the ticket itself.
"""
from __future__ import annotations

from SupportOpsBot.runtime import prompts
from SupportOpsBot.runtime.setups.base import Setup, Turn, TurnResult
from SupportOpsBot.runtime.shared.flow import PartFlow
from SupportOpsBot.runtime.shared.strategies import Agent
from SupportOpsBot.runtime.shared.tool_catalog import GATHER_TOOLS


class SingleGraph(Setup):
    agents, control = "single", "graph"

    def run(self, turn: Turn) -> TurnResult:
        agent = Agent("agent", "agent", prompts.load("agent", today=turn.today, facts=turn.facts(),
                                                     control=prompts.load("control", "graph")),
                      turn.policy.planner_context())
        flow = PartFlow(turn.ctx, self.strategy["agent"], agent, turn.registry.toolset(GATHER_TOOLS),
                        turn.registry.writes)
        part = flow.run(turn.brief)
        return TurnResult(part.status, part.note)
