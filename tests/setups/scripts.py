"""Turn one scenario into the LLM script any setup needs, so the same behaviour is tested in all 24 variants.

A scenario says what the working agent (the single agent, or the specialist the supervisor asks) reads,
which writes it makes and how it ends. The script differs only where the setups differ: agentic agents
make writes and finish with tools; graph agents gather, then propose; react takes one call per turn,
plan_execute plans, runs and is asked once more.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from support_ops.setups import build_setup
from support_ops.shared.config import load_config
from support_ops.shared.pipeline import TurnOutput, run_turn
from support_ops.shared.policy.rag import RagPolicy
from support_ops.shared.rules import Rules
from tests.fakes import ORDER, FakeDatabase, FakeRetriever, ScriptedLLM, as_json, clock, plan, text, tool_call


def policy_read(mode: str) -> tuple[str, dict[str, Any]]:
    if mode == "rag":
        return "policy", {"action": "search", "query": "refund for a returned item"}
    return "policy", {"action": "read", "skill": "returns-and-refunds", "file": "refunds.md"}


def citation(mode: str) -> str:
    return "REF-Refunds-6.1" if mode == "rag" else "refunds.md"


@dataclass
class Scenario:
    reads: list[tuple[str, dict[str, Any]]]
    writes: list[dict[str, Any]] = field(default_factory=list)   # backend_write arguments
    review: bool = False             # the writes need request_review first (agentic)
    outcome: str = "resolved"
    note: str = "Done."
    specialist: str = "returns_refunds"
    request: str = "Handle the customer's request."


def refund(amount: float, item: str, mode: str) -> dict[str, Any]:
    return {"operation": "refund", "order_id": ORDER, "amount": amount, "items": [{"item_id": item, "quantity": 1}],
            "reason": "damaged_item", "cited_clauses": [citation(mode)]}


def _calls(llm: ScriptedLLM, role: str, strategy: str, steps: list[tuple[str, dict[str, Any]]],
           last: list[tuple[str, dict[str, Any]]], end_text: str | None = None) -> None:
    """`steps`, then `last` (the finishing calls); react ends with `end_text` when nothing finishes it."""
    if strategy == "react":
        llm.push(role, *(tool_call(name, **args) for name, args in steps + last))
        if end_text is not None:
            llm.push(role, text(end_text))
    elif last:
        llm.push(role, plan(*steps), plan(*last))   # the plan, then the completion check
    else:
        llm.push(role, plan(*steps), plan())


def worker_script(llm: ScriptedLLM, cfg: dict[str, Any], scenario: Scenario, role: str, strategy: str) -> None:
    if cfg["setup"]["control"] == "agentic":
        writes = [call for write in scenario.writes for call in
                  ([("request_review", {**write, "review_reason": "The policy and the records support it."})]
                   if scenario.review else []) + [("backend_write", write)]]
        _calls(llm, role, strategy, scenario.reads, [*writes, ("finish", {"outcome": scenario.outcome,
                                                                          "note": scenario.note})])
    else:
        _calls(llm, role, strategy, scenario.reads, [], end_text="Gathered.")
        llm.push(role, as_json({"outcome": scenario.outcome, "note": scenario.note, "writes": scenario.writes,
                                **({"review_reason": "The policy and the records support it."}
                                   if scenario.writes else {})}))


def script(setup_id: str, scenario: Scenario) -> ScriptedLLM:
    cfg, llm = load_config(setup=setup_id), ScriptedLLM()
    strategy = cfg["strategy"]
    if cfg["setup"]["agents"] == "single":
        worker_script(llm, cfg, scenario, "agent", strategy["agent"])
        return llm
    worker_script(llm, cfg, scenario, "specialist", strategy["specialists"])
    ask = [("ask_specialist", {"specialist": scenario.specialist, "request": scenario.request})]
    finish = [("finish", {"outcome": scenario.outcome, "note": scenario.note})] \
        if cfg["setup"]["control"] == "agentic" else []
    if strategy["supervisor"] == "react":
        llm.push("supervisor", *(tool_call(name, **args) for name, args in ask + finish))
        if not finish:
            llm.push("supervisor", text("Every part has been asked."))
    else:
        llm.push("supervisor", plan(*ask), plan(*finish))
    return llm


def mode(setup_id: str) -> str:
    return load_config(setup=setup_id)["policy"]["mode"]


def run(setup_id: str, llm: ScriptedLLM, message: str = "Please refund my order.", *,
        db: FakeDatabase | None = None, **kwargs: Any) -> tuple[TurnOutput, FakeDatabase]:
    cfg = load_config(setup=setup_id)
    rules = Rules.load(cfg["policy"]["rules_file"])
    default = RagPolicy(FakeRetriever(), rules) if cfg["policy"]["mode"] == "rag" else None
    policy = kwargs.pop("policy", default)
    db = db if db is not None else FakeDatabase()
    output = run_turn(build_setup(cfg=cfg), llm, ticket_id="T1", message=message, reader=db, writer=db,
                      policy=policy, clock=clock(), **kwargs)
    return output, db
