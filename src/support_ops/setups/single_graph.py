"""S5 (RAG) and S6 (skills): one agent inside a fixed LangGraph sequence decided in code.

The agent gathers facts with read and policy tools, then proposes; the graph checks the rules, has any
write reviewed when required, and writes (shared/flow.py). The agent cannot write, start a review or end
the ticket itself.
"""
from __future__ import annotations

from support_ops import prompts
from support_ops.setups.base import Setup, Turn, TurnResult
from support_ops.shared.flow import PartFlow
from support_ops.shared.strategies import Agent
from support_ops.shared.tools import GATHER_TOOLS


class SingleGraph(Setup):
    agents, control = "single", "graph"

    def run(self, turn: Turn) -> TurnResult:
        agent = Agent("agent", "agent", prompts.load("agent", today=turn.today, facts=turn.facts(),
                                                     control=prompts.load("control", "graph")),
                      turn.policy.planner_context())
        flow = PartFlow(turn.ctx, self.strategy["agent"], agent, turn.registry.toolset(GATHER_TOOLS),
                        turn.gate, turn.executor)
        part = flow.run(turn.brief)
        return TurnResult(part.status, part.note)
