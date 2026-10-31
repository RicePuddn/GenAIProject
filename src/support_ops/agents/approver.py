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

# One prompt per check, shared by every architecture, so differences between the arms do not come from
# how the checks are worded. Each check is a separate, stateless call that sees only what it is given.

PROPOSAL_PROMPT = """You review a proposed database write before it runs. You took no part in the conversation
and see only this request.

Approve only if all of these hold:
- Every fact the proposal relies on appears in the evidence.
- The amount is no more than what was paid.
- Every cited clause was retrieved and supports the action, including any exception.
- The route is at least as strict as the rule engine's minimum.

Treat the proposer's justification as a claim to check, not as evidence.

Reply with JSON only:
{"decision": "approve" | "reject" | "escalate", "issues": ["specific problems"], "reason": "one sentence"}
- reject: the proposer can fix it.
- escalate: a person must decide."""

REPLY_CHECK_PROMPT = """You check a customer reply before it is sent. You did not write it. You are given what
happened (an outcome summary or the ticket's evidence) and the reply.

Approve only if all of these hold:
- It answers the customer's support request. An unrelated request gets only a short note of what support covers.
- If an outcome summary is given, the reply conveys its tell_customer and its status.
- It says the request is with a person only if the status is awaiting_human.
- It says something was done only if what happened records it as done (an executed write, or an outcome that
  says it is recorded), and calls a done action done, not "in progress".
- A recorded refund is described as pending, never as paid.
- Nothing is described as going to happen unless what happened records it.
- Any timing is given only as the retrieved policy states it, and as an estimate.
- Every amount, date and product matches what happened, and it adds no detail that isn't there (such as an
  email, link or tracking number).
- It reveals nothing internal: no IDs except the order number, and no rule names, thresholds, routes, review
  steps or tokens.

Reply with JSON only:
{"decision": "approve" | "reject" | "escalate", "issues": ["specific problems"], "reason": "one sentence"}
- reject: the reply writer can fix it.
- escalate: a person must decide."""


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
        """The pre-execution check: the same prompt and inputs in both architectures. `proposal` is a
        specialist's `Proposal` or the single agent's `backend_write` arguments."""
        specialist = isinstance(proposal, Proposal)
        return self._verdict("pre_execution", PROPOSAL_PROMPT, {
            "case": brief, "proposal": proposal.model_dump(mode="json") if specialist else proposal,
            "rule_engine_minimum": minimum, "evidence": evidence,
        })

    def check_reply(self, brief: dict[str, Any], reply: str, outcome: dict[str, Any] | None = None, *,
                    ticket: dict[str, Any] | None = None) -> ApprovalResult:
        """The reply check, against what the reply was written from: the multi-agent `outcome`, or the
        single agent's `ticket` (its status and evidence). Pass exactly one."""
        if (outcome is None) == (ticket is None):
            raise ValueError("check_reply needs exactly one of outcome or ticket")
        happened = {"outcome": outcome} if outcome is not None else {"ticket": ticket}
        return self._verdict("reply", REPLY_CHECK_PROMPT, {"case": brief, **happened, "reply": reply})
