"""`policy` tool: wraps a retriever so the agent can search the policy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from support_ops.helpers.config import ROOT
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
        """Load models and build the index now rather than on the first search."""
        ...


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
        policy = cfg["policy"]
        folder = Path(policy["corpus_dir"]) / policy["corpus_size"]
        folder = (folder if folder.is_absolute() else ROOT / folder).resolve()
        return cls(shared_retriever(str(folder), RagSettings.from_config(policy.get("rag", {}))), rules)

    def warm_up(self) -> None:
        self.retriever.warm_up()

    def planner_context(self) -> str:
        return ("Policy sections (contents must still be searched with tools): "
                + json.dumps(self.retriever.sections()))

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        if args.get("action") == "check":
            return self.rules.check(args.get("operation", ""), args.get("amount"))
        if args.get("action") == "search":
            return {"query": args["query"], "clauses": self.retriever.search(args["query"])}
        raise ValueError("unsupported policy action")
