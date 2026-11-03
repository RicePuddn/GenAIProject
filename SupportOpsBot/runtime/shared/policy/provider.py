"""The contract both policy modes (RAG and skills) meet, and the choice between them from `policy.mode`."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from SupportOpsBot.runtime.shared.rules import Rules
    from SupportOpsBot.services.structure.structure import ToolArguments


class PolicyProvider(Protocol):
    arguments: type[ToolArguments]  # argument model of the `policy` tool
    description: str                # `policy` tool description the LLM reads
    prompt_hint: str                # the one policy-specific sentence in the agent prompts

    def planner_context(self) -> str:
        """Policy overview given to a plan_execute planner up front ("" for none)."""
        ...

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute validated `policy` tool arguments."""
        ...

    def citable(self, result: dict[str, Any]) -> set[str]:
        """What a successful `run` result lets the agent cite: clause IDs, or the files it read."""
        ...

    def warm_up(self) -> None:
        """Prepare policy retrieval before ticket timing begins."""
        ...

    def scoped(self, scope: Sequence[str]) -> PolicyProvider:
        """The same tool for a specialist, keeping by default to `scope` (RAG: section names; skills: skill
        folders), with scope all reaching the rest."""
        ...


def policy_from_config(cfg: dict[str, Any], rules: Rules) -> PolicyProvider:
    mode = cfg["policy"]["mode"]
    if mode == "skills":
        from SupportOpsBot.runtime.shared.policy.skills import SkillsPolicy
        return SkillsPolicy(cfg["policy"]["skills_dir"], rules)
    if mode == "rag":
        from SupportOpsBot.runtime.shared.policy.rag import RagPolicy
        return RagPolicy.from_config(cfg, rules)
    raise ValueError(f"unsupported policy.mode: {mode!r}")
