"""What every setup implements and returns. A setup implements only its body: the pipeline runs the guard
before it, and the reply, the reply check and the ticket save after it, the same way for all of them."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from SupportOpsBot.runtime import prompts
from SupportOpsBot.runtime.shared.context import TurnContext
from SupportOpsBot.runtime.shared.gates.write_gate import WriteExecutor, WriteGate
from SupportOpsBot.runtime.shared.policy.provider import PolicyProvider
from SupportOpsBot.runtime.shared.strategies import AgentRun
from SupportOpsBot.runtime.shared.tools import ToolRegistry

Status = Literal["resolved", "awaiting_customer", "awaiting_human", "in_progress"]   # in_progress: the turn was cut off
STRICTNESS: tuple[Status, ...] = ("resolved", "awaiting_customer", "awaiting_human")


@dataclass
class TurnResult:
    """How the setup's body ended. The evidence and the writes are in the turn's context."""
    status: Status
    note: str | None = None   # resolved: what was done; awaiting_customer: the question; awaiting_human: why
    hold: bool = False        # send the fixed hold reply with no further LLM call (guard flag, budget)
    parts: list[dict[str, Any]] = field(default_factory=list)   # multi-agent: each specialist's report


@dataclass
class Turn:
    """What a setup's body gets: the turn's context and the shared tools and gates for this ticket."""
    ctx: TurnContext
    brief: dict[str, Any]     # the customer's message, earlier turns and signed-in account
    registry: ToolRegistry
    gate: WriteGate
    executor: WriteExecutor
    policy: PolicyProvider
    customer_id: str | None

    def facts(self, policy: PolicyProvider | None = None) -> str:
        """How every working agent checks facts and policy (prompts/facts.md), with `policy`'s hint (by
        default the turn's own provider)."""
        signed_in = "\n" + prompts.load("facts", "signed_in", account=self.customer_id) if self.customer_id else ""
        return prompts.load("facts", "facts", policy_hint=(policy or self.policy).prompt_hint,
                            signed_in=signed_in)

    @property
    def today(self) -> str:
        return self.ctx.clock.now().date().isoformat()


class Setup(ABC):
    """One of the four code paths. `cfg` is a resolved config: its setup must match this class."""
    agents: ClassVar[str]    # single or multi
    control: ClassVar[str]   # agentic or graph

    def __init__(self, cfg: dict[str, Any]):
        setup = cfg["setup"]
        if (setup["agents"], setup["control"]) != (self.agents, self.control):
            raise ValueError(f"{setup['id']} is {setup['agents']}/{setup['control']}, not {self.agents}/{self.control}")
        self.cfg, self.strategy = cfg, cfg["strategy"]

    @abstractmethod
    def run(self, turn: Turn) -> TurnResult:
        """The setup's body for one turn."""


def finished(turn: Turn, run: AgentRun, parts: list[dict[str, Any]] | None = None) -> TurnResult:
    """An agentic setup ends where its agent called finish, unless a gate stopped the turn first."""
    if turn.ctx.stopped:
        return TurnResult("awaiting_human", turn.ctx.stop_reason, parts=parts or [])
    if run.finished is None:
        return TurnResult("awaiting_human", run.error or "the agent stopped without calling finish", parts=parts or [])
    return TurnResult(run.finished["outcome"], run.finished["note"], parts=parts or [])
