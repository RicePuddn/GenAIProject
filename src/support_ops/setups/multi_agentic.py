"""S3 (RAG) and S4 (skills): a supervisor and two specialists, all choosing their own steps.

The supervisor asks a specialist for each part and finishes when it decides to. Each specialist holds its
subset of the data tools plus backend_write and request_review, and acts on its part as the single agent
does (S1/S2), finishing with a report back to the supervisor.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from support_ops.setups.base import STRICTNESS, Setup, Turn, TurnResult, finished
from support_ops.setups.multi_common import (SPECIALIST_TOOLS, ask_tool, report, specialist, specialist_brief,
                                             strictest, supervisor)
from support_ops.shared.strategies import run_agent
from support_ops.shared.tools import WRITE_TOOLS, Toolset, finish_tool


class MultiAgentic(Setup):
    agents, control = "multi", "agentic"

    def run(self, turn: Turn) -> TurnResult:
        parts: list[dict[str, Any]] = []

        def ask(name: str, request: str) -> dict[str, Any]:
            start = len(turn.ctx.evidence)
            agent, policy = specialist(turn, name, "agentic")
            tools = turn.registry.toolset((*SPECIALIST_TOOLS[name], *WRITE_TOOLS), extra=[finish_tool()],
                                          terminal="finish", policy=policy)
            run = run_agent(self.strategy["specialists"], turn.ctx, agent, tools, specialist_brief(turn, request))
            writes = [item["result"] for item in turn.ctx.evidence[start:] if item["tool"] == "backend_write"
                      and item["result"].get("status") == "executed"]
            if run.finished is None:
                return {**report("awaiting_human", None, writes),
                        "error": run.error or "the specialist stopped without calling finish"}
            return report(run.finished["outcome"], run.finished["note"], writes)

        tools = Toolset([ask_tool(ask, parts), finish_tool()], terminal="finish")
        run = run_agent(self.strategy["supervisor"], turn.ctx, supervisor(turn, "agentic"), tools, turn.brief)
        result = finished(turn, run, parts)
        # The supervisor cannot end the ticket milder than a specialist reported, as in S7/S8: a part that
        # needs a person or the customer is never closed as resolved.
        floor = strictest(parts)
        if result.status in STRICTNESS and STRICTNESS.index(floor) > STRICTNESS.index(result.status):
            notes = [part.get("note") or part.get("error") for part in parts if part["outcome"] == floor]
            turn.ctx.log("outcome_raised", finished=result.status, outcome=floor)
            result = replace(result, status=floor, note=" ".join(filter(None, notes)) or result.note)
        return result
