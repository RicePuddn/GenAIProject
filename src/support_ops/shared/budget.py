"""Per-ticket budgets and the retry policy, the same for every setup.

One `Budget` covers the whole ticket, shared by every agent in it: a multi-agent setup's supervisor and
specialists draw on the same agent steps and tool calls (`limits`) as the single agent, not on their own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from support_ops.shared.tracing import Tracer


class BudgetExceeded(Exception):
    """A per-ticket limit was reached; the ticket goes to a human."""


@dataclass
class Budget:
    """`limits` in config. Checked before each agent step, tool call and gate LLM call."""
    limits: dict[str, Any]
    tracer: Tracer
    tool_calls: int = 0
    agent_steps: int = 0

    def _check_cost(self) -> None:
        if self.tracer.cost_usd >= self.limits["max_cost_usd"]:
            raise BudgetExceeded("LLM cost budget reached")

    def agent_step(self) -> None:
        """Before an LLM call by the agent, the supervisor or a specialist."""
        if self.agent_steps >= self.limits["max_agent_steps"]:
            raise BudgetExceeded("agent step limit reached")
        self._check_cost()
        self.agent_steps += 1

    def tool_call(self) -> None:
        """Before a data tool runs, a write included."""
        if self.tool_calls >= self.limits["max_tool_calls"]:
            raise BudgetExceeded("tool call limit reached")
        self._check_cost()
        self.tool_calls += 1

    def gate_call(self) -> None:
        """Before an LLM call by a gate (the write review, the reply writer or the reply check)."""
        self._check_cost()

    @property
    def tool_calls_left(self) -> int:
        return self.limits["max_tool_calls"] - self.tool_calls


@dataclass(frozen=True)
class Retries:
    """`retries` in config."""
    tool_errors: int     # re-runs of a tool that failed for a reason the model can't fix
    invalid_json: int    # extra tries for a malformed JSON reply
    backoff_s: float     # wait before re-running a tool, times the attempt number

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Retries:
        values = cfg["retries"]
        retries = cls(tool_errors=values["tool_errors"], invalid_json=values["invalid_json"],
                      backoff_s=values["backoff_s"])
        if retries.tool_errors < 0 or retries.invalid_json < 0 or retries.backoff_s < 0:
            raise ValueError("retries values must not be negative")
        return retries
