"""`policy` tool: wraps a retriever so the agent can search the policy.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from support_ops.helpers.arguments import NonEmpty, ToolArguments
from support_ops.policy.provider import ScopedPolicyTool
from support_ops.policy.rag.retriever import RagSettings, shared_retriever
from support_ops.policy.rules import Rules
from support_ops.systems.tools import RagPolicyArguments


class Retriever(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]:
        """Most relevant clauses first, each with id, title, section, source, text, score and via."""
        ...

    def sections(self) -> list[str]:
        """Section names in corpus order."""
        ...

    def warm_up(self) -> None:
        """Load and exercise models before serving tickets."""
        ...


class PolicySearchArguments(ToolArguments):
    """Retrieve the policy clauses most relevant to a query, with their exceptions. scope own (the
    default) keeps clauses from your policy sections; use all when the case crosses sections."""
    action: Literal["search"] = "search"
    query: NonEmpty
    scope: Literal["own", "all"] = "own"


# Compare policy titles and stored domain names using the same format.
def section_key(section: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", section.casefold()).strip("-").removesuffix("-policy")


def own_sections(result: dict[str, Any], sections: Sequence[str]) -> dict[str, Any]:
    """Keep search hits in the specialist's scope and their direct references."""
    wanted = {section_key(section) for section in sections}
    clauses = result.get("clauses", [])
    kept_ids = {clause["id"] for clause in clauses
                if clause["via"] == "search" and section_key(clause["section"]) in wanted}
    references = {ref for clause in clauses if clause["id"] in kept_ids for ref in clause.get("references", [])}
    kept = [clause for clause in clauses if clause["id"] in kept_ids
            or clause["id"] in references]
    filtered = {**result, "clauses": kept, "scope": "own", "sections": list(sections)}
    if dropped := len(clauses) - len(kept):
        filtered["filtered_out"] = dropped
    if not kept:
        filtered["note"] = ("No clause from your sections matched. If the case crosses into another area, "
                            "search again with scope all.")
    return filtered


class RagPolicy:
    arguments = RagPolicyArguments
    description = RagPolicyArguments.__doc__
    prompt_hint = ("Search the policy for the relevant clauses before advising. "
                   "Never invent policy terms or missing facts.")
    scope_hint = ("Policy search covers your own policy sections; search with scope all when the case crosses "
                  "into another area.")
    scope_setting = "policy_sections"
    strict_scope = False   # a section missing from the corpus is logged per run instead

    def __init__(self, retriever: Retriever, rules: Rules | None = None):
        self.retriever = retriever
        self.rules = rules or Rules.load()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], rules: Rules) -> RagPolicy:
        policy = cfg["policy"]
        settings = RagSettings.from_config(policy.get("rag", {}))
        return cls(shared_retriever(settings), rules)

    # Preload the same cached retriever later used by the agent tool.
    def warm_up(self) -> None:
        self.retriever.warm_up()

    def planner_context(self) -> str:
        return ("Policy sections (contents must still be searched with tools): "
                + json.dumps(self.retriever.sections()))

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        if args.get("action") == "check":
            return self.rules.check(args.get("operation", ""), args.get("amount"), args.get("facts"))
        if args.get("action") == "search":
            return {"query": args["query"], "clauses": self.retriever.search(args["query"])}
        raise ValueError("unsupported policy action")

    def scoped_tool(self, scope: Sequence[str] | None) -> ScopedPolicyTool:
        """Search only, limited by default to the specialist's `scope` sections. The retriever's top-k is
        filtered afterwards; a native section filter in the retriever would return k clauses from the
        section instead (see docs/multi-agent.md, vector DB notes)."""
        def run(ticket_id: str, args: dict[str, Any]) -> dict[str, Any]:
            result = self.run({"action": "search", "query": args["query"]})
            if args.get("scope", "own") == "all" or not scope:
                return {**result, "scope": "all"}
            return own_sections(result, scope)

        return ScopedPolicyTool(PolicySearchArguments, PolicySearchArguments.__doc__, run,
                                self.prompt_hint, self.scope_hint, self.planner_context,
                                lambda result: {clause["id"] for clause in result.get("clauses", [])})

    def check_scope(self, scope: Sequence[str] | None) -> list[str]:
        try:
            known = {section_key(section) for section in self.retriever.sections()}
        except Exception:   # the corpus cannot be read: nothing to compare with
            return []
        return [section for section in scope or [] if known and section_key(section) not in known]
