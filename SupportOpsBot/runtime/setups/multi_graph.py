"""S7 (RAG) and S8 (skills): a supervisor and two specialists inside fixed stages decided in code.

The supervisor only delegates: it asks a specialist for each part, by its strategy (one part at a time,
or all parts planned first). There is no triage: it does not classify or redirect the message. Each part
runs the same fixed sequence as S5/S6 (shared/flow.py), with the specialist in the gather stage. The
ticket ends on the strictest outcome of its parts, or with a person if the supervisor gave up. A message
with nothing for the store asks no specialist and ends resolved, answered by the reply.
"""
from __future__ import annotations

from typing import Any

from SupportOpsBot.runtime.setups.base import Setup, Turn, TurnResult
from SupportOpsBot.runtime.setups.multi_common import (SPECIALIST_TOOLS, ask_tool, report, specialist, specialist_brief,
                                             strictest, supervisor)
from SupportOpsBot.runtime.shared.flow import PartFlow
from SupportOpsBot.runtime.shared.strategies import run_agent
from SupportOpsBot.runtime.shared.tools import Toolset


class MultiGraph(Setup):
    agents, control = "multi", "graph"

    def run(self, turn: Turn) -> TurnResult:
        parts: list[dict[str, Any]] = []

        def ask(name: str, request: str) -> dict[str, Any]:
            agent, policy = specialist(turn, name, "graph")
            flow = PartFlow(turn.ctx, self.strategy["specialists"], agent,
                            turn.registry.toolset(SPECIALIST_TOOLS[name], policy=policy), turn.registry.writes)
            part = flow.run(specialist_brief(turn, request))
            return report(part.status, part.note, part.writes)

        run = run_agent(self.strategy["supervisor"], turn.ctx, supervisor(turn, "graph"),
                        Toolset([ask_tool(ask, parts)]), turn.brief)
        if turn.ctx.stopped:
            return TurnResult("awaiting_human", turn.ctx.stop_reason, parts=parts)
        # A supervisor that gave up may have left parts unasked, so the parts that ran cannot end the ticket.
        if run.error:
            return TurnResult("awaiting_human", f"the supervisor stopped: {run.error}", parts=parts)
        status = strictest(parts)
        notes = [part["note"] for part in parts if part["outcome"] == status and part.get("note")]
        return TurnResult(status, " ".join(notes) or None, parts=parts)
