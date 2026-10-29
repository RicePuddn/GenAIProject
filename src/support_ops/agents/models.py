"""Typed messages passed between the multi-agent roles.

The supervisor emits a `Triage`, a specialist emits a `Proposal`, the approver emits an
`ApprovalResult`, and a human route ends in a `HandoffPacket`. Agents produce these as JSON;
validation failures go back to the agent as errors, never into a write.
"""
from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from support_ops.policy.routing import Route

NonEmpty = Annotated[str, Field(min_length=1)]
Specialist = Literal["orders_shipping", "returns_refunds"]
Action = Literal["none", "refund", "return", "exchange", "order_change", "account_change"]
WRITE_ACTIONS = {"refund", "return", "exchange", "order_change", "account_change"}
GOODS_ACTIONS = {"return", "exchange"}   
ITEM_ACTIONS = GOODS_ACTIONS | {"refund"}  
SPECIALIST_ACTIONS: dict[str, frozenset[str]] = {
    "orders_shipping": frozenset({"none", "order_change"}),
    "returns_refunds": frozenset({"none", "refund", "return", "exchange"}),
}


class AgentOutput(BaseModel):
    # Not strict: models write JSON numbers and strings loosely, but unknown fields are errors.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Triage(AgentOutput):
    """The supervisor's first read of a customer message."""
    intent: NonEmpty                       # e.g. order_status, refund_request, return_request
    urgency: Literal["low", "normal", "high"]
    language: NonEmpty                     # e.g. en
    specialist: Specialist | None          # None: no specialist fits, so hand off
    duplicate_of: str | None = None        # open ticket ID this message repeats
    wants_human: bool = False              # the customer asked for a person
    off_topic: bool = False                # nothing to do with the store's support topics: redirect, no handoff
    confidence: float = Field(ge=0, le=1)
    reason: NonEmpty


class ItemLine(AgentOutput):
    """One order line coming back."""
    item_id: NonEmpty = Field(validation_alias=AliasChoices("item_id", "order_item_id", "id"),
                              description="The order item ID from get_items_for_order, e.g. ITM-0031-2.")
    quantity: int = Field(default=1, ge=1, description="How many units of that line come back.")


class ProposalDraft(AgentOutput):
    """A specialist's recommended action, as the model writes it (the `route` tool's arguments).
    Proposing never executes anything. The field descriptions are what the model sees."""
    action: Action = Field(description="none for information only or a question to the customer.")
    order_id: NonEmpty | None = Field(default=None, description="Required for every action except none.")
    amount: float | None = Field(default=None, gt=0, allow_inf_nan=False,
                                 description="Refunds only: the amount in the order's currency.")
    items: list[ItemLine] = Field(default_factory=list, description=(
        "Returns and exchanges: the order lines coming back (required). Refunds: the order lines the "
        "refund is for; the amount may not exceed what was paid for them. A refund without items needs "
        "the approver."))
    reason_code: NonEmpty | None = Field(default=None, description="e.g. damaged_item, late_delivery.")
    cited_clauses: list[NonEmpty] = Field(default_factory=list,
                                          description="Clause IDs you retrieved that support the action.")
    policy_exception: bool = Field(default=False,
                                   description="True when the action relies on an exception to the main rule.")
    route: Route = Field(description="auto, approver or human; at least as strict as the rules allow.")
    reason: NonEmpty = Field(description="Why this action and route, for the approver and operator.")
    customer_question: NonEmpty | None = Field(
        default=None, description="With action none: the question only the customer can answer.")

    @model_validator(mode="after")
    def required_for_action(self) -> Self:
        # Fields an action does not use are dropped rather than rejected: they change nothing.
        if self.action != "refund":
            self.amount = None
        if self.action not in ITEM_ACTIONS:
            self.items = []
        if self.action in WRITE_ACTIONS and not self.order_id:
            raise ValueError(f"{self.action} requires order_id")
        if self.action == "refund" and self.amount is None:
            raise ValueError("refund requires amount")
        if self.action in GOODS_ACTIONS and not self.items:
            raise ValueError(f"{self.action} requires items: the order item IDs and quantities coming back")
        if len({line.item_id for line in self.items}) != len(self.items):
            raise ValueError("list each order item once")
        if self.action in WRITE_ACTIONS and not self.cited_clauses:
            raise ValueError(f"{self.action} requires at least one cited policy clause")
        return self


class Proposal(ProposalDraft):
    """A draft attributed to the specialist that made it (set by code, not the model)."""
    specialist: Specialist


class ApprovalResult(AgentOutput):
    """The approver's verdict, before execution (proposal) or after it (reply)."""
    stage: Literal["pre_execution", "reply"]
    decision: Literal["approve", "reject", "escalate"]
    issues: list[NonEmpty] = Field(default_factory=list)
    reason: NonEmpty

    @model_validator(mode="after")
    def issues_explain_a_rejection(self) -> Self:
        if self.decision != "approve" and not self.issues:
            raise ValueError(f"{self.decision} requires at least one issue")
        return self


class HandoffPacket(AgentOutput):
    """What a human operator receives when a ticket moves to awaiting_human."""
    ticket_id: NonEmpty
    reason: NonEmpty                       # the trigger: route, guard flag, budget, customer request...
    summary: NonEmpty
    actions_taken: list[NonEmpty] = Field(default_factory=list)
    cited_clauses: list[NonEmpty] = Field(default_factory=list)
    recommendation: NonEmpty
    proposal: Proposal | None = None
    min_route: dict | None = None          # the rule engine's result, when a proposal was checked
