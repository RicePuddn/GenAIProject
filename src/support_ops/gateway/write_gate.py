"""Whether a write may run, kept apart from running it.

Shared by every arm with no approval gateway between its agent and the database: the single agent's
`backend_write` tool today, and the deterministic flows' graph nodes next. `WriteGate` decides and
never writes: the recorded facts and the rules, the citations, and on the approver route an
independent review. `WriteExecutor` writes only what it is handed, once per idempotency key. Who calls
them, and when, is the architecture's choice: the single agent's tool runs them in turn when the agent
writes; a graph can make each one a node.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable

from support_ops.backend.postgres import PostgresWriter, WriteRefused
from support_ops.gateway.ticket_state import TicketState
from support_ops.policy.citations import unread_citations
from support_ops.policy.facts import WRITE_OPERATIONS, BackendFacts

# Takes {"proposal", "rule_engine_minimum", "evidence"}; returns a decision (approve, reject or
# escalate), issues and reason.
Reviewer = Callable[[dict[str, Any]], dict[str, Any]]


def handoff_args(args: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"action": "handoff", "operation": args["operation"], "order_id": args["order_id"],
            "amount": args.get("amount"), "reason": reason}


class WriteGate:
    def __init__(self, facts: BackendFacts, ticket_state: TicketState, reviewer: Reviewer | None = None,
                 max_rejections: int = 1):
        self.facts, self.ticket_state = facts, ticket_state
        # Without a reviewer, writes requiring review go to a human.
        self.reviewer = reviewer
        self.max_rejections = max_rejections   # review.max_rejections: rejections the agent may fix
        # Per ticket: successful read and policy results, what they let the agent cite, and review rejections.
        self.evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.citable: dict[str, set[str]] = defaultdict(set)
        self.rejections: dict[str, int] = defaultdict(int)

    def record(self, ticket_id: str, tool: str, result: dict[str, Any], citable: set[str]) -> None:
        """A successful read or policy result: evidence for the review, and what it lets the agent cite."""
        self.evidence[ticket_id].append({"tool": tool, "result": result})
        self.citable[ticket_id] |= citable

    def check(self, ticket_id: str, args: dict[str, Any]) -> dict[str, Any]:
        """The recorded facts, the rules and the citations. Returns status `allowed` with the rule check
        (whose route says whether `review` must run next), or the result to give the agent instead: a
        handoff, a block or a request for review_reason. Citation mistakes raise ValueError, which the
        agent can fix."""
        operation, order_id = args["operation"], args["order_id"]
        if operation not in WRITE_OPERATIONS:
            return self.ticket_state.run(ticket_id, handoff_args(args, "unsupported write operation"))
        item_ids = sorted(line["item_id"] for line in args.get("items", []))
        decision = self.ticket_state.rules.check(operation, args.get("amount"),
                                                 self.facts.read(operation, order_id, item_ids))
        if decision["requires_human"]:
            return self.ticket_state.run(ticket_id, handoff_args(args, decision["reason"]))
        if not decision["allowed"]:
            return {"status": "blocked", "reason": decision["reason"], "policy_check": decision}
        # Every write rests on retrieved policy, whatever its route, as every multi-agent write proposal does.
        cited = args.get("cited_clauses", [])
        if not cited:
            raise ValueError(f"{operation} requires at least one cited policy clause: retrieve the policy "
                             "that supports it and list it in cited_clauses")
        if decision["route"] == "approver" and not args.get("review_reason", "").strip():
            return {"status": "review_required", "policy_check": decision,
                    "reason": "This write goes to an independent reviewer: supply review_reason explaining "
                              "how the cited policy and recorded facts support it."}
        if unread := unread_citations(cited, self.citable[ticket_id]):
            raise ValueError(f"cited_clauses {unread} were not returned by the policy tool in this ticket; "
                             "retrieve them first, or cite only the clause IDs or files you retrieved")
        return {"status": "allowed", "policy_check": decision}

    def review(self, ticket_id: str, args: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any] | None:
        """The independent review of a write on the approver route: None when it approves, else the
        result to give the agent. A rejection the agent can fix comes back as a block; an escalation, or
        a rejection after max_rejections, goes to a human."""
        if self.reviewer is None:
            return self.ticket_state.run(ticket_id, handoff_args(args, "no reviewer for a write requiring review"))
        verdict = self.reviewer({"proposal": args, "rule_engine_minimum": decision,
                                 "evidence": self.evidence[ticket_id]})
        if verdict["decision"] == "approve":
            return None
        issues = "; ".join(verdict.get("issues") or []) or verdict.get("reason", "")
        self.rejections[ticket_id] += 1
        if verdict["decision"] != "reject" or self.rejections[ticket_id] > self.max_rejections:
            return self.ticket_state.run(ticket_id, handoff_args(args, f"reviewer {verdict['decision']}: {issues}"))
        return {"status": "blocked", "reason": f"the independent review rejected this write: {issues}",
                "review": verdict, "policy_check": decision}


class WriteExecutor:
    def __init__(self, writer: PostgresWriter, ticket_state: TicketState):
        self.writer, self.ticket_state = writer, ticket_state

    @staticmethod
    def _request(ticket_id: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        operation, order_id = args["operation"], args["order_id"]
        lines = sorted(args.get("items", []), key=lambda line: line["item_id"])
        request = {"ticket_id": ticket_id, "operation": WRITE_OPERATIONS[operation], "order_id": order_id,
                   "amount": args.get("amount"), "items": lines}
        # The amount is left out of the key, so one ticket cannot write the same action for two amounts.
        return request, f"{ticket_id}:{operation}:{order_id}:{','.join(line['item_id'] for line in lines)}"

    def replay(self, ticket_id: str, args: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any] | None:
        """The first result of a write this ticket already made, or None. A conflicting repeat raises."""
        request, key = self._request(ticket_id, args)
        if (previous := self.writer.replay(key)) is None:
            return None
        if json.dumps(previous["request"], sort_keys=True) != json.dumps(request, sort_keys=True):
            raise ValueError("conflicting amount or items for an existing write")
        return {**previous["result"], "replayed": True, "policy_check": decision}

    def apply(self, ticket_id: str, args: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        """Write what the gate allowed, and audit it."""
        request, key = self._request(ticket_id, args)
        try:
            result = self.writer.apply(request, idempotency_key=key, reason=args["reason"],
                                       at=self.ticket_state.clock.now(), payment_id=decision.get("payment_id"),
                                       return_id=decision.get("return_id"))
        except WriteRefused as exc:
            return {"status": "blocked", "reason": str(exc), "policy_check": decision}
        self.ticket_state.run(ticket_id, {"action": "audit", "type": "write", "operation": args["operation"],
                                          "order_id": args["order_id"], "amount": args.get("amount"),
                                          "write_id": result["write_id"], "review_reason": args.get("review_reason"),
                                          "cited_clauses": args.get("cited_clauses", [])})
        return {**result, "policy_check": decision}
