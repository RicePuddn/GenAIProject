"""Approver / QA agent: a second check before execution and a check of every reply.

Its verdict is advisory to the graph but binding on tokens: the gateway issues none on the
approver route without an approve from `check_proposal`. Unreadable output counts as escalate.
The single agent uses the same two checks, each one separate LLM call: `check_proposal` before
`backend_write` executes a write requiring review, and `check_reply` on every final reply.
"""
from __future__ import annotations

import json
from typing import Any

from support_ops.agents.models import ApprovalResult, Proposal
from support_ops.agents.runtime import AgentContext, chat_json

VERDICT_FORMAT = (' Return JSON only: {"decision": "approve" | "reject" | "escalate", '
                  '"issues": ["specific problems"], "reason": "one sentence"}. ')
VERDICT = VERDICT_FORMAT + "reject: the specialist can fix it; escalate: an operator must decide."

# The bar every pre-execution review holds a proposal to, in both architectures.
PROPOSAL_CRITERIA = (
    "Approve only if every fact it relies on appears in the tool evidence, the amount does not exceed "
    "what was paid, the cited clauses were retrieved and support the action (including any exception), "
    "and the rule engine's minimum route is respected."
)

PROPOSAL_PROMPT = (
    "You are the approver for a policy-bound support team, giving the second check on a specialist's "
    "proposal before anything is executed. " + PROPOSAL_CRITERIA + VERDICT
)

# The single agent's review: a separate, stateless call that sees the customer's request, the proposed
# write and its evidence, never the agent's own messages or reasoning, so it does not inherit them.
WRITE_REVIEW_PROMPT = (
    "You are an independent reviewer for a policy-bound support service. A support agent proposes the "
    "database write below; you took no part in the conversation and see only this request. Treat the "
    "agent's justification as a claim to verify against the evidence, not as evidence. "
    + PROPOSAL_CRITERIA + VERDICT_FORMAT
    + "reject: the agent can fix it; escalate: an operator must decide."
)

# What no customer reply may reveal, in both architectures.
REPLY_INTERNALS = ("it reveals nothing internal: item IDs, policy clause IDs, rule names, thresholds, routes, "
                   "internal review steps or tokens.")

REPLY_CHECK_PROMPT = (
    "You check a customer reply before it is sent. Approve only if it conveys the outcome's "
    "tell_customer and status (nothing described as pending review unless the status is "
    "awaiting_human, nothing described as done unless tell_customer says it is recorded), every "
    "amount, date, product and promise matches the outcome, it mentions no detail absent from the "
    "outcome (such as an email, link or tracking number), and " + REPLY_INTERNALS + VERDICT
)

# The single agent's reply check. Its replies are written from the tool evidence, not from an outcome
# summary, so the same checks are made against the evidence.
EVIDENCE_REPLY_CHECK_PROMPT = (
    "You check a customer reply before it is sent; you took no part in writing it. Approve only if it "
    "answers the customer's support request from the ticket's evidence (nothing described as pending "
    "review or referred unless the status is awaiting_human, nothing described as done unless the "
    "evidence has a backend_write with status executed, a recorded refund described as pending and never "
    "as paid), every amount, date, product and promise matches the evidence, it mentions no detail absent "
    "from the evidence (such as an email, link or tracking number), and " + REPLY_INTERNALS
    + " A request unrelated to support may only get a short note of what support covers."
    + VERDICT_FORMAT + "reject: the reply writer can fix it; escalate: an operator must decide."
)


class Approver:
    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def _verdict(self, stage: str, system: str, payload: dict[str, Any]) -> ApprovalResult:
        self.ctx.log("node_start", node="approver", stage=stage)
        def with_stage(data: Any) -> dict[str, Any]:
            if not isinstance(data, dict):
                raise ValueError("verdict must be a JSON object")
            return {**data, "stage": stage}   # the stage is set by code, not the model

        try:
            result = chat_json(self.ctx, "approver", [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ], ApprovalResult, prepare=with_stage)
        except ValueError as exc:
            result = ApprovalResult(stage=stage, decision="escalate", issues=[f"unreadable approver output: {exc}"],
                                    reason="The approver's verdict could not be read.")
        self.ctx.log("approval", **result.model_dump(mode="json"))
        return result

    def check_proposal(self, brief: dict[str, Any], proposal: Proposal | dict[str, Any], minimum: dict[str, Any],
                       evidence: list[dict[str, Any]]) -> ApprovalResult:
        """The pre-execution check, the same inputs in both architectures. A specialist's `Proposal` gets
        the team approver's prompt; the single agent's `backend_write` arguments get the independent
        reviewer's, with the same criteria."""
        specialist = isinstance(proposal, Proposal)
        return self._verdict("pre_execution", PROPOSAL_PROMPT if specialist else WRITE_REVIEW_PROMPT, {
            "case": brief, "proposal": proposal.model_dump(mode="json") if specialist else proposal,
            "rule_engine_minimum": minimum, "evidence": evidence,
        })

    def check_reply(self, brief: dict[str, Any], reply: str, outcome: dict[str, Any] | None = None, *,
                    ticket: dict[str, Any] | None = None) -> ApprovalResult:
        """The reply check, against what the reply was written from: the multi-agent `outcome`, or the
        single agent's `ticket` (its status and evidence). Pass exactly one."""
        if (outcome is None) == (ticket is None):
            raise ValueError("check_reply needs exactly one of outcome or ticket")
        if outcome is not None:
            return self._verdict("reply", REPLY_CHECK_PROMPT, {"case": brief, "outcome": outcome, "reply": reply})
        return self._verdict("reply", EVIDENCE_REPLY_CHECK_PROMPT, {"case": brief, "ticket": ticket, "reply": reply})
