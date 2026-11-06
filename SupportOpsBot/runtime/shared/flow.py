"""The graph setups' fixed sequence of stages for one case, as a LangGraph (S5–S8).

    gather → propose → check → review (approver route only) → write → next write … → done
      ^                  |         |                            |
      +------------------+---------+---- feedback --------------+

gather: the agent works with read and policy tools only, by its strategy. propose: one structured
output, the `Proposal`. check, review and write use the shared write coordinator, the same operations
used by agentic named write tools, for each proposed write in turn. A block, a fixable rejection or a refused
citation goes back to gather with the reason; a handoff ends the turn. The agent never writes, starts a
review or ends the case itself: code decides each next stage.
"""
from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph

from SupportOpsBot.runtime import prompts
from SupportOpsBot.runtime.shared.context import TurnContext, chat_json
from SupportOpsBot.runtime.shared.gates.write_gate import WriteCoordinator
from SupportOpsBot.runtime.shared.strategies import Agent, run_agent
from SupportOpsBot.runtime.shared.tools import (Toolset, run_with_retries,
                                                validate_arguments, begin_data_call, end_data_call)
from SupportOpsBot.runtime.shared.tool_catalog import GRAPH_TRACE_TOOLS
from SupportOpsBot.services.structure.structure import (BackendWriteArguments, PartResult, PartState, Proposal,
                                                        ProposedLine, ProposedWrite)

class PartFlow:
    def __init__(self, ctx: TurnContext, strategy: str, agent: Agent, tools: Toolset, writes: WriteCoordinator):
        self.ctx, self.strategy, self.agent, self.tools = ctx, strategy, agent, tools
        self.writes = writes
        self.brief: dict[str, Any] = {}
        graph = StateGraph(PartState)
        nodes = ("gather", "propose", "check", "review", "write")
        for name in nodes:
            graph.add_node(name, getattr(self, f"_{name}"))
            graph.add_conditional_edges(name, lambda state: state["next"], {**{node: node for node in nodes},
                                                                           "done": END})
        graph.add_edge(START, "gather")
        self.graph = graph.compile()

    def run(self, brief: dict[str, Any]) -> PartResult:
        """Carry the case from gathering to its last write. The budget ends a loop of refusals."""
        self.brief, start = brief, len(self.ctx.evidence)
        state = self.graph.invoke({"feedback": None, "proposal": None, "pending": [], "decision": None,
                                   "attempt": None}, {"recursion_limit": 200})
        writes = [item["result"] for item in self.ctx.evidence[start:] if item["tool"] in GRAPH_TRACE_TOOLS
                  and item["result"].get("status") == "executed"]
        if self.ctx.stopped:
            return PartResult("awaiting_human", self.ctx.stop_reason, writes)
        proposal: Proposal = state["proposal"]
        return PartResult(proposal.outcome, proposal.note, writes)

    def _own_evidence(self) -> list[dict[str, Any]]:
        return [{"tool": item["tool"], "arguments": item["arguments"], "result": item["result"]}
                for item in self.ctx.evidence if item["agent"] == self.agent.name]

    def _back(self, reason: str) -> dict[str, Any]:
        self.ctx.log("proposal_refused", agent=self.agent.name, reason=reason)
        return {"feedback": reason, "pending": [], "decision": None, "attempt": None, "next": "gather"}

    # ------------------------------------------------------------- stages

    def _gather(self, state: PartState) -> dict[str, Any]:
        self.ctx.log("node_start", node="gather", agent=self.agent.name)
        brief = dict(self.brief)
        if state["feedback"]:
            brief.update(evidence=self._own_evidence(), feedback=state["feedback"])
        run = run_agent(self.strategy, self.ctx, self.agent, self.tools, brief)
        if run.error:   # the facts may be incomplete, so no proposal can rest on them
            self.ctx.stop(f"{self.agent.name} could not gather the facts: {run.error}")
        return {"next": "done" if self.ctx.stopped else "propose"}

    def _propose(self, state: PartState) -> dict[str, Any]:
        self.ctx.log("node_start", node="propose", agent=self.agent.name)
        try:
            proposal = chat_json(self.ctx, self.agent.role, [
                {"role": "system", "content": self.agent.prompt + "\n\n" + prompts.load(
                    "propose")},
                {"role": "user", "content": json.dumps({**self.brief, "evidence": self._own_evidence(),
                                                        "feedback": state["feedback"]}, default=str)},
            ], Proposal, before_call=self.ctx.budget.agent_step)
        except ValueError as exc:
            self.ctx.stop(f"no valid proposal: {exc}")
            return {"next": "done"}
        self.ctx.log("proposal", agent=self.agent.name, **proposal.model_dump(mode="json"))
        if proposal.outcome == "awaiting_human":
            self.ctx.stop(proposal.note)
            return {"proposal": proposal, "next": "done"}
        pending = []
        for index, write in enumerate(proposal.writes, start=1):
            try:
                args = validate_arguments(BackendWriteArguments, write.model_dump(mode="json", exclude_none=True))
            except ValueError as exc:
                return {"proposal": proposal, **self._back(f"write {index} is invalid: {exc}")}
            pending.append({**args, **({"review_reason": proposal.review_reason} if proposal.review_reason else {})})
        return {"proposal": proposal, "pending": pending, "feedback": None, "next": "check" if pending else "done"}

    def _check(self, state: PartState) -> dict[str, Any]:
        args = state["pending"][0]
        attempt = self._begin(args)

        def replay_or_check() -> dict[str, Any]:
            return self.writes.prepare(args)

        checked = run_with_retries(self.ctx, attempt["call_id"], "backend_write", args, replay_or_check)
        if checked.get("status") != "allowed":
            return self._end(attempt, args, checked, state)
        decision = checked["policy_check"]
        return {"attempt": attempt, "decision": decision,
                "next": "review" if decision["route"] == "approver" else "write"}

    def _review(self, state: PartState) -> dict[str, Any]:
        args = state["pending"][0]
        if refused := self.writes.review(args, state["decision"]):
            return self._end(state["attempt"], args, refused, state)
        return {"next": "write"}

    def _write(self, state: PartState) -> dict[str, Any]:
        args = state["pending"][0]
        written = run_with_retries(self.ctx, state["attempt"]["call_id"], "backend_write", args,
                                   lambda: self.writes.apply(args, state["decision"]))
        return self._end(state["attempt"], args, written, state)

    # ------------------------------------------------------------- graph write accounting

    def _begin(self, args: dict[str, Any]) -> dict[str, Any]:
        """A graph write attempt keeps the established internal `backend_write` trace label."""
        return begin_data_call(self.ctx, self.agent.name, "backend_write", args, node="check")

    def _end(self, attempt: dict[str, Any], args: dict[str, Any], value: dict[str, Any],
             state: PartState) -> dict[str, Any]:
        end_data_call(self.ctx, self.agent.name, "backend_write", args, value, attempt)
        if self.ctx.stopped:
            return {"next": "done"}
        if value.get("status") != "executed":
            return self._back(value.get("error") or value.get("reason") or value.get("status", "refused"))
        rest = state["pending"][1:]
        return {"pending": rest, "decision": None, "attempt": None, "next": "check" if rest else "done"}
