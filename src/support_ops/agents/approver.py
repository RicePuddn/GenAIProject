"""Approver / QA agent: a second check before execution and a check of every reply.

Its verdict is advisory to the graph but binding on tokens: the gateway issues none on the
approver route without an approve from `check_proposal`. Unreadable output counts as escalate.
"""
from __future__ import annotations

import json
from typing import Any

from support_ops.agents.models import ApprovalResult, Proposal
from support_ops.agents.runtime import AgentContext, chat_json

VERDICT = (' Return JSON only: {"decision": "approve" | "reject" | "escalate", '
           '"issues": ["specific problems"], "reason": "one sentence"}. '
           "reject: the specialist can fix it; escalate: an operator must decide.")

PROPOSAL_PROMPT = (
    "You are the approver for a policy-bound support team, giving the second check on a specialist's "
    "proposal before anything is executed. Approve only if every fact it relies on appears in the tool "
    "evidence, the amount does not exceed what was paid, the cited clauses were retrieved and support "
    "the action (including any exception), and the rule engine's minimum route is respected." + VERDICT
)

REPLY_CHECK_PROMPT = (
    "You check a customer reply before it is sent. Approve only if it conveys the outcome's "
    "tell_customer and status (nothing described as pending review unless the status is "
    "awaiting_human, nothing described as done unless tell_customer says it is recorded), every "
    "amount, date, product and promise matches the outcome, it mentions no detail absent from the "
    "outcome (such as an email, link or tracking number), and it reveals nothing internal: item IDs, "
    "policy clause IDs, rule names, thresholds, routes, internal review steps or tokens." + VERDICT
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

    def check_proposal(self, brief: dict[str, Any], proposal: Proposal, minimum: dict[str, Any],
                       evidence: list[dict[str, Any]]) -> ApprovalResult:
        return self._verdict("pre_execution", PROPOSAL_PROMPT, {
            "case": brief, "proposal": proposal.model_dump(mode="json"),
            "rule_engine_minimum": minimum, "evidence": evidence,
        })

    def check_reply(self, brief: dict[str, Any], reply: str, outcome: dict[str, Any]) -> ApprovalResult:
        return self._verdict("reply", REPLY_CHECK_PROMPT, {"case": brief, "outcome": outcome, "reply": reply})
