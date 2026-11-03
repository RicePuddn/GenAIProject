"""The approver: the independent review of a write before it runs, and the check of every reply.

Each check is a separate, stateless LLM call with one prompt for every setup (prompts/approver.md). It
sees the customer's request and the evidence, never an agent's own messages or reasoning, so it does not
inherit them. Unreadable output counts as escalate.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from support_ops import prompts
from support_ops.shared.context import TurnContext, chat_json


class ApprovalResult(BaseModel):
    """The approver's verdict on a write (`write`) or a reply (`reply`)."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    stage: Literal["write", "reply"]
    decision: Literal["approve", "reject", "escalate"]
    issues: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    reason: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def issues_explain_a_rejection(self) -> Self:
        if self.decision != "approve" and not self.issues:
            raise ValueError(f"{self.decision} requires at least one issue")
        return self


class Approver:
    def __init__(self, ctx: TurnContext):
        self.ctx = ctx

    def _verdict(self, stage: Literal["write", "reply"], payload: dict[str, Any]) -> ApprovalResult:
        def with_stage(data: Any) -> dict[str, Any]:
            if not isinstance(data, dict):
                raise ValueError("verdict must be a JSON object")
            return {**data, "stage": stage}   # set by code, not the model

        try:
            result = chat_json(self.ctx, "approver", [
                {"role": "system", "content": prompts.load("approver", stage)},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ], ApprovalResult, before_call=self.ctx.budget.gate_call, prepare=with_stage)
        except ValueError as exc:
            result = ApprovalResult(stage=stage, decision="escalate", issues=[f"unreadable approver output: {exc}"],
                                    reason="The approver's verdict could not be read.")
        self.ctx.log("approval", **result.model_dump(mode="json"))
        return result

    def check_write(self, brief: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        """The write gate's reviewer: `request` holds the write, its rule check and the evidence."""
        return self._verdict("write", {"case": brief, **request}).model_dump(mode="json")

    def check_reply(self, brief: dict[str, Any], ticket: dict[str, Any], reply: str) -> ApprovalResult:
        return self._verdict("reply", {"case": brief, "ticket": ticket, "reply": reply})
