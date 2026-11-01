"""The graph setups' fixed sequence of stages for one case, as a LangGraph (S5–S8).

    gather → propose → check → review (approver route only) → write → next write … → done
      ^                  |         |                            |
      +------------------+---------+---- feedback --------------+

gather: the agent works with read and policy tools only, by its strategy. propose: one structured
output, the `Proposal`. check, review and write: the write gate and executor, the same code an agentic
setup's backend_write runs, for each proposed write in turn. A block, a fixable rejection or a refused
citation goes back to gather with the reason; a handoff ends the turn. The agent never writes, starts a
review or ends the case itself: code decides each next stage.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from support_ops import prompts
from support_ops.shared.context import TurnContext, chat_json
from support_ops.shared.gates.write_gate import WriteExecutor, WriteGate
from support_ops.shared.strategies import Agent, run_agent
from support_ops.shared.tools import BackendWriteArguments, Toolset, validate_arguments

NonEmpty = Annotated[str, Field(min_length=1)]
# Keys a model may add for its own reasoning; never part of a proposal.
REASONING_KEYS = frozenset({"analysis", "reasoning", "thinking", "thoughts", "thought"})


class AgentOutput(BaseModel):
    # Not strict: models write JSON numbers and strings loosely, but unknown fields are errors.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProposedLine(AgentOutput):
    item_id: NonEmpty = Field(validation_alias=AliasChoices("item_id", "order_item_id", "id"),
                              description="The order item ID from get_items_for_order.")
    quantity: int = Field(default=1, ge=1)


class ProposedWrite(AgentOutput):
    """One write, with backend_write's arguments."""
    operation: NonEmpty = Field(description="refund, return_request or cancel.")
    order_id: NonEmpty
    amount: float | None = Field(default=None, gt=0, allow_inf_nan=False, description="Refunds only.")
    items: list[ProposedLine] = Field(default_factory=list)
    reason: NonEmpty = Field(description="Why, in a short phrase such as damaged_item or order_cancellation.")
    cited_clauses: list[NonEmpty] = Field(default_factory=list)
    policy_exception: bool = Field(default=False, description="True when it relies on an exception to the main rule.")
    route: Literal["approver", "human"] | None = Field(default=None, description=(
        "Stricter handling than the rules require: approver for a review, human for an operator to decide."))


class Proposal(AgentOutput):
    """What the agent proposes once the facts are gathered. Proposing writes nothing."""
    outcome: Literal["resolved", "awaiting_customer", "awaiting_human"]
    note: NonEmpty = Field(description=(
        "resolved: what to tell the customer; awaiting_customer: the question to ask; awaiting_human: why."))
    writes: list[ProposedWrite] = Field(default_factory=list)
    review_reason: NonEmpty | None = None

    @model_validator(mode="before")
    @classmethod
    def tidy_model_output(cls, data: Any) -> Any:
        """Undo three habits of models writing JSON: reasoning written into the object (gpt-oss puts its
        analysis channel there in JSON mode on some providers), the proposal nested one level down, and
        null for an empty list."""
        if not isinstance(data, dict):
            return data
        data = {key: value for key, value in data.items() if key not in REASONING_KEYS}
        nested = [value for value in data.values() if isinstance(value, dict) and "outcome" in value]
        if "outcome" not in data and len(nested) == 1:
            data = {key: value for key, value in nested[0].items() if key not in REASONING_KEYS}
        return {key: value for key, value in data.items() if not (key == "writes" and value is None)}


@dataclass
class PartResult:
    status: Literal["resolved", "awaiting_customer", "awaiting_human"]
    note: str | None
    writes: list[dict[str, Any]] = field(default_factory=list)


class PartState(TypedDict, total=False):
    feedback: str | None                 # why the last proposal was refused, for the next gather
    proposal: Proposal | None
    pending: list[dict[str, Any]]        # validated writes still to carry out, the first in progress
    decision: dict[str, Any] | None      # the rule check of the write in progress
    attempt: dict[str, Any] | None       # the write in progress as a backend_write call: id and start time
    next: str


class PartFlow:
    def __init__(self, ctx: TurnContext, strategy: str, agent: Agent, tools: Toolset, gate: WriteGate,
                 executor: WriteExecutor):
        self.ctx, self.strategy, self.agent, self.tools = ctx, strategy, agent, tools
        self.gate, self.executor = gate, executor
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
        writes = [item["result"] for item in self.ctx.evidence[start:] if item["tool"] == "backend_write"
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
        run_agent(self.strategy, self.ctx, self.agent, self.tools, brief)
        return {"next": "done" if self.ctx.stopped else "propose"}

    def _propose(self, state: PartState) -> dict[str, Any]:
        self.ctx.log("node_start", node="propose", agent=self.agent.name)
        try:
            proposal = chat_json(self.ctx, self.agent.role, [
                {"role": "system", "content": self.agent.prompt + "\n\n" + prompts.load(
                    "propose", schema=json.dumps(Proposal.model_json_schema()))},
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
        try:
            if (previous := self.executor.replay(args)) is not None:
                return self._end(attempt, args, previous, state)
            checked = self.gate.check(args)
        except ValueError as exc:
            return self._end(attempt, args, {"error": str(exc)}, state)
        if checked["status"] != "allowed":
            return self._end(attempt, args, checked, state)
        decision = checked["policy_check"]
        return {"attempt": attempt, "decision": decision,
                "next": "review" if decision["route"] == "approver" else "write"}

    def _review(self, state: PartState) -> dict[str, Any]:
        args = state["pending"][0]
        if refused := self.gate.review(args, state["decision"]):
            return self._end(state["attempt"], args, refused, state)
        return {"next": "write"}

    def _write(self, state: PartState) -> dict[str, Any]:
        args = state["pending"][0]
        return self._end(state["attempt"], args, self.executor.apply(args, state["decision"]), state)

    # ------------------------------------------------------------- a write as a backend_write call

    def _begin(self, args: dict[str, Any]) -> dict[str, Any]:
        """A write attempt counts and is traced as one backend_write call, as in the agentic setups."""
        call_id = self.ctx.next_call_id()
        self.ctx.log("node_start", node="check", agent=self.agent.name)
        self.ctx.log("tool_call_requested", call_id=call_id, tool="backend_write", input=args, agent=self.agent.name)
        self.ctx.budget.tool_call()
        self.ctx.log("tool_start", call_id=call_id, tool="backend_write", input=args, agent=self.agent.name)
        return {"call_id": call_id, "started": time.perf_counter()}

    def _end(self, attempt: dict[str, Any], args: dict[str, Any], value: dict[str, Any],
             state: PartState) -> dict[str, Any]:
        self.ctx.log("tool_end", call_id=attempt["call_id"], tool="backend_write", agent=self.agent.name,
                     elapsed_s=round(time.perf_counter() - attempt["started"], 6), output=value)
        self.ctx.evidence.append({"agent": self.agent.name, "tool": "backend_write", "arguments": args,
                                  "result": value})
        if self.ctx.stopped:
            return {"next": "done"}
        if value.get("status") != "executed":
            return self._back(value.get("error") or value.get("reason") or value.get("status", "refused"))
        rest = state["pending"][1:]
        return {"pending": rest, "decision": None, "attempt": None, "next": "check" if rest else "done"}
