"""The contract each policy-retrieval modes (RAG/skills) share"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from pydantic import BaseModel

from support_ops.helpers.arguments import ToolArguments


@dataclass(frozen=True)
class ScopedPolicyTool:
    """A multi-agent specialist's `policy` tool: the provider's own search or browsing, limited by default to
    the specialist's part of the policy, without `check` (there the `route` tool applies the hard rules).
    It carries the prompt text that goes with it, so the workflow never needs to know the mode."""
    arguments: type[BaseModel]
    description: str
    run: Callable[[str, dict[str, Any]], dict[str, Any]]   # (ticket_id, validated arguments)
    prompt_hint: str        # how to use the tool, as in the single agent
    scope_hint: str         # the sentence about the specialist's scope and scope all
    context: Callable[[], str] = lambda: ""   # overview for the plan-and-execute planner, built when asked
    # What a successful result lets the specialist cite: clause IDs, or the files it read.
    citable: Callable[[dict[str, Any]], set[str]] = lambda result: set()

    def planner_context(self) -> str:
        return self.context()


class PolicyProvider(Protocol):
    arguments: type[ToolArguments]  # argument model of the `policy` tool
    description: str                # `policy` tool description the LLM reads
    prompt_hint: str                # the one policy-specific sentence in the system prompt
    scope_setting: str              # multi_agent setting that lists each specialist's scope
    strict_scope: bool              # a scope naming missing parts fails the workflow at start-up

    def planner_context(self) -> str:
        """Policy overview given to the plan-and-execute planner up front ("" for none)."""
        ...

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute validated `policy` tool arguments."""
        ...

    def warm_up(self) -> None:
        """Prepare policy retrieval before ticket timing begins."""
        ...

    def scoped_tool(self, scope: Sequence[str] | None) -> ScopedPolicyTool:
        """The multi-agent specialist's tool for `scope` (None: the whole policy)."""
        ...

    def check_scope(self, scope: Sequence[str] | None) -> list[str]:
        """Names in `scope` the policy does not have; never raises."""
        ...
