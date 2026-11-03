"""`policy` tool: wraps a retriever so the agent can search the policy.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Literal, Protocol, Self

from pydantic import Field, model_validator

from support_ops.shared.policy.rag.retriever import RagSettings, shared_retriever
from support_ops.shared.rules import Rules
from support_ops.shared.tools import CHECK_DESCRIPTION, NonEmpty, ToolArguments


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


class RagPolicyArguments(ToolArguments):
    __doc__ = ("search: retrieve the policy clauses most relevant to a query, with their exceptions; "
               + CHECK_DESCRIPTION)
    action: Literal["search", "check"]
    query: NonEmpty | None = None
    operation: Literal["delivery"] | None = None
    order_id: NonEmpty | None = Field(default=None, description="The order whose parcels a delivery check covers.")

    @model_validator(mode="after")
    def required_for_action(self) -> Self:
        if self.action == "search" and not self.query:
            raise ValueError("search requires query")
        if self.action == "check" and not (self.operation and self.order_id):
            raise ValueError("check requires operation delivery and order_id")
        return self


class ScopedRagArguments(RagPolicyArguments):
    __doc__ = (RagPolicyArguments.__doc__ + " scope own (the default) keeps search hits from your policy "
               "sections; use all when the case crosses sections.")
    scope: Literal["own", "all"] = "own"


# Compare policy titles and stored section names in the same form.
def section_key(section: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", section.casefold()).strip("-").removesuffix("-policy")


def own_sections(result: dict[str, Any], sections: Sequence[str]) -> dict[str, Any]:
    """Keep search hits in the specialist's sections, and the clauses they directly reference."""
    wanted = {section_key(section) for section in sections}
    clauses = result.get("clauses", [])
    kept_ids = {clause["id"] for clause in clauses
                if clause["via"] == "search" and section_key(clause["section"]) in wanted}
    references = {ref for clause in clauses if clause["id"] in kept_ids for ref in clause.get("references", [])}
    kept = [clause for clause in clauses if clause["id"] in kept_ids or clause["id"] in references]
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

    def __init__(self, retriever: Retriever, rules: Rules | None = None):
        self.retriever = retriever
        self.rules = rules or Rules.load()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], rules: Rules) -> RagPolicy:
        settings = RagSettings.from_config(cfg["policy"]["rag"])
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

    @staticmethod
    def citable(result: dict[str, Any]) -> set[str]:
        """A search can be cited by the IDs of the clauses it returned, references included."""
        return {clause["id"] for clause in result.get("clauses", [])}

    def scoped(self, scope: Sequence[str]) -> ScopedRagPolicy:
        """The same tool for a specialist: by default only search hits from its own sections. The top-k is
        filtered afterwards; a section filter in the retriever would return k clauses from the section."""
        return ScopedRagPolicy(self, tuple(scope))


class ScopedRagPolicy:
    arguments = ScopedRagArguments
    description = ScopedRagArguments.__doc__

    def __init__(self, policy: RagPolicy, scope: tuple[str, ...]):
        self.policy, self.scope = policy, scope
        self.prompt_hint = (policy.prompt_hint + " Policy search covers your own policy sections; search with "
                            "scope all when the case crosses into another area.")
        self.citable, self.warm_up, self.planner_context = policy.citable, policy.warm_up, policy.planner_context

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self.policy.run({key: value for key, value in args.items() if key != "scope"})
        if args["action"] != "search":
            return result
        return {**result, "scope": "all"} if args.get("scope", "own") == "all" else own_sections(result, self.scope)
