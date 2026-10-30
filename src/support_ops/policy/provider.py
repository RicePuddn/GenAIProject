"""The contract each policy-retrieval modes (RAG/skills) share"""
from __future__ import annotations

from typing import Any, Protocol

from support_ops.helpers.arguments import ToolArguments


class PolicyProvider(Protocol):
    arguments: type[ToolArguments]  # argument model of the `policy` tool
    description: str                # `policy` tool description the LLM reads
    prompt_hint: str                # the one policy-specific sentence in the system prompt

    def planner_context(self) -> str:
        """Policy overview given to the plan-and-execute planner up front ("" for none)."""
        ...

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute validated `policy` tool arguments."""
        ...

    def warm_up(self) -> None:
        """Prepare policy retrieval before ticket timing begins."""
        ...
