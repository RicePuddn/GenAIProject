"""The one write path: whether a write may run, kept apart from running it.

Every setup's writes come here. `WriteGate` decides and never writes: the recorded facts and the rules,
then the citations (each must be a clause or file a policy search or read returned in this ticket), then,
on the approver route, the independent review. `WriteExecutor` writes only what it is handed, once per
idempotency key. An agentic setup reaches both through the `backend_write` and `request_review` tools; a
graph setup calls them as nodes. Who asks for the review is the control style: the agent, with
`request_review`, or the graph.
"""
from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Callable

from support_ops.shared.context import TurnContext
from support_ops.shared.db import PostgresWriter, WriteRefused
from support_ops.shared.gates.ticket import TicketLog
from support_ops.shared.policy.citations import unread_citations
from support_ops.shared.policy.provider import PolicyProvider
from support_ops.shared.rules import WRITE_OPERATIONS, BackendFacts, Rules

# Takes {"write", "rule_check", "evidence"}; returns a decision (approve, reject or escalate), issues and reason.
Reviewer = Callable[[dict[str, Any]], dict[str, Any]]
WRITE_TOOLS = {"backend_write", "request_review"}


def done(decision: dict[str, Any]) -> dict[str, Any]:
    """The rule check as reported with a write that is done. Its reason describes the state before the
    write (e.g. "the actual update is still required"), which would read as if it had not happened."""
    return {key: value for key, value in decision.items() if key != "reason"}


class WriteGate:
    def __init__(self, ctx: TurnContext, facts: BackendFacts, rules: Rules, policy: PolicyProvider,
                 log: TicketLog, reviewer: Reviewer):
        self.ctx, self.facts, self.rules, self.policy, self.log = ctx, facts, rules, policy, log
        self.reviewer = reviewer
        self.max_rejections = ctx.cfg["review"]["max_rejections"]   # rejections the agent may fix
        self.rejections = 0
        self.approved: set[str] = set()   # writes request_review approved, by approval_key

    def evidence(self) -> list[dict[str, Any]]:
        """The successful reads and policy results of this turn: what the reviewer checks a write against."""
        return [{"tool": item["tool"], "result": item["result"]} for item in self.ctx.evidence
                if item["tool"] not in WRITE_TOOLS and "error" not in item["result"]]

    def citable(self) -> set[str]:
        return {cited for item in self.ctx.evidence if item["tool"] == "policy" and "error" not in item["result"]
                for cited in self.policy.citable(item["result"])}

    def handoff(self, args: dict[str, Any], reason: str) -> dict[str, Any]:
        """A person must decide: the turn ends here, in every setup."""
        self.log.audit("write_handoff", operation=args["operation"], order_id=args["order_id"],
                       amount=args.get("amount"), reason=reason)
        self.ctx.stop(reason)
        return {"status": "awaiting_human", "reason": reason}

    def check(self, args: dict[str, Any]) -> dict[str, Any]:
        """The recorded facts, the rules and the citations. Returns status `allowed` with the rule check
        (whose route says whether a review must come next), or the result to give the agent instead: a
        handoff or a block. Citation mistakes raise ValueError, which the agent can fix."""
        operation, order_id = args["operation"], args["order_id"]
        if operation not in WRITE_OPERATIONS:
            return self.handoff(args, "unsupported write operation")
        item_ids = sorted(line["item_id"] for line in args.get("items", []))
        decision = self.rules.check(operation, args.get("amount"), self.facts.read(operation, order_id, item_ids))
        if decision["requires_human"]:
            return self.handoff(args, decision["reason"])
        if not decision["allowed"]:
            return {"status": "blocked", "reason": decision["reason"], "policy_check": decision}
        cited = args.get("cited_clauses", [])
        if not cited:
            raise ValueError(f"{operation} requires at least one cited policy clause: retrieve the policy "
                             "that supports it and list it in cited_clauses")
        if unread := unread_citations(cited, self.citable()):
            raise ValueError(f"cited_clauses {unread} were not returned by the policy tool in this ticket; "
                             "cite only clause IDs or files a policy search or read returned. A policy check "
                             "names its rule's sources without their text, so they cannot be cited: search "
                             "or read the policy for them first")
        return {"status": "allowed", "policy_check": decision}

    def review(self, args: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any] | None:
        """The independent review of a write on the approver route: None when it approves, else the result
        to give the agent. A rejection the agent can fix comes back as a block; an escalation, or a
        rejection after review.max_rejections, goes to a human."""
        self.ctx.log("node_start", node="write_review")
        verdict = self.reviewer({"write": args, "rule_check": decision, "evidence": self.evidence()})
        self.ctx.log("write_review", decision=verdict["decision"], issues=verdict.get("issues", []))
        if verdict["decision"] == "approve":
            return None
        issues = "; ".join(verdict.get("issues") or []) or verdict.get("reason", "")
        self.rejections += 1
        if verdict["decision"] != "reject" or self.rejections > self.max_rejections:
            return self.handoff(args, f"reviewer {verdict['decision']}: {issues}")
        return {"status": "blocked", "reason": f"the independent review rejected this write: {issues}",
                "review": verdict, "policy_check": decision}

    def request_review(self, args: dict[str, Any]) -> dict[str, Any]:
        """An agent asks for the review itself: the same checks as the write, then the independent review.
        An approval covers only this exact write."""
        checked = self.check(args)
        if checked["status"] != "allowed":
            return checked
        decision = checked["policy_check"]
        if decision["route"] != "approver":
            return {"status": "not_required", "policy_check": decision,
                    "reason": "This write needs no review: write it with backend_write."}
        if refused := self.review(args, decision):
            return refused
        self.approved.add(self.approval_key(args))
        return {"status": "approved", "policy_check": decision,
                "reason": "Approved: write it with backend_write, with the same operation, order, amount and items."}

    def is_approved(self, args: dict[str, Any]) -> bool:
        return self.approval_key(args) in self.approved

    @staticmethod
    def approval_key(args: dict[str, Any]) -> str:
        """What an approval is bound to, so a different amount or different items need their own review."""
        amount = args.get("amount")
        lines = sorted(f"{line['item_id']}x{line.get('quantity', 1)}" for line in args.get("items", []))
        return json.dumps([args["operation"], args["order_id"],
                           None if amount is None else f"{Decimal(str(amount)):.2f}", lines])


class WriteExecutor:
    def __init__(self, ctx: TurnContext, writer: PostgresWriter, log: TicketLog):
        self.ctx, self.writer, self.log = ctx, writer, log

    def _request(self, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        ticket_id, operation, order_id = self.ctx.ticket_id, args["operation"], args["order_id"]
        lines = sorted(args.get("items", []), key=lambda line: line["item_id"])
        request = {"ticket_id": ticket_id, "operation": WRITE_OPERATIONS[operation], "order_id": order_id,
                   "amount": args.get("amount"), "items": lines}
        # The amount is left out of the key, so one ticket cannot write the same action for two amounts.
        return request, f"{ticket_id}:{operation}:{order_id}:{','.join(line['item_id'] for line in lines)}"

    def replay(self, args: dict[str, Any]) -> dict[str, Any] | None:
        """The first result of a write this ticket already made, or None. A conflicting repeat raises.
        Checked before the rules, which would now see the write's own effect (e.g. an order already
        cancelled)."""
        if args["operation"] not in WRITE_OPERATIONS:
            return None
        request, key = self._request(args)
        if (previous := self.writer.replay(key)) is None:
            return None
        if json.dumps(previous["request"], sort_keys=True) != json.dumps(request, sort_keys=True):
            raise ValueError("conflicting amount or items for an existing write")
        return {**previous["result"], "replayed": True}

    def apply(self, args: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        """Write what the gate allowed, and audit it."""
        request, key = self._request(args)
        try:
            result = self.writer.apply(request, idempotency_key=key, reason=args["reason"],
                                       at=self.ctx.clock.now(), payment_id=decision.get("payment_id"),
                                       return_id=decision.get("return_id"))
        except WriteRefused as exc:
            return {"status": "blocked", "reason": str(exc), "policy_check": decision}
        self.log.audit("write", operation=args["operation"], order_id=args["order_id"], amount=args.get("amount"),
                       write_id=result["write_id"], review_reason=args.get("review_reason"),
                       cited_clauses=args.get("cited_clauses", []))
        self.ctx.log("write_result", result=result)
        return {**result, "policy_check": done(decision)}
