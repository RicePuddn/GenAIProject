"""`policy` tool: wraps a retriever so the agent can search the policy.
"""
from __future__ import annotations

import json
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
