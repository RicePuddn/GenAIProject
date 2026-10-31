"""Typed messages passed between the multi-agent roles.

The supervisor emits a `Triage`, a specialist emits a `Proposal`, the approver emits an
`ApprovalResult`, and a human route ends in a `HandoffPacket`. Agents produce these as JSON;
validation failures go back to the agent as errors, never into a write.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from support_ops.backend.postgres import TicketCategory, TicketPriority
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


class Task(AgentOutput):
    """One part of the customer's message, for one specialist."""
    specialist: Specialist
    request: NonEmpty                      # exactly what this specialist handles; name the order IDs


class Case(AgentOutput):
    """The supervisor's reading of the message. These are the customer's claims, not verified facts."""
    issue: NonEmpty                        # what is wrong or wanted, in a sentence
    intent: NonEmpty                       # e.g. order_status, refund_request, return_request
    category: TicketCategory               # the support ticket's category
    urgency: TicketPriority
    language: NonEmpty                     # e.g. en
    order_ids: list[NonEmpty] = Field(default_factory=list)
    items: list[NonEmpty] = Field(default_factory=list)     # products or item IDs the customer named
    amounts: list[NonEmpty] = Field(default_factory=list)   # amounts the customer named, e.g. SGD 32


class SupervisorDecision(AgentOutput):
    """What the supervisor does next. Code follows it; the gates after a specialist still apply."""
    thought: NonEmpty                      # the reasoning, including what still has to happen
    action: Literal["delegate", "handoff", "redirect", "finish"]
    tasks: list[Task] = Field(default_factory=list)   # delegate: in the order the customer raised them
    reason: NonEmpty                       # handoff: what the operator is told
    case: Case | None = None

    @model_validator(mode="after")
    def tasks_match_action(self) -> Self:
        if self.action != "delegate":
            self.tasks = []
        elif not self.tasks:
            raise ValueError("delegate requires tasks")
        if len({task.specialist for task in self.tasks}) != len(self.tasks):
            raise ValueError("list each specialist once in tasks, with everything it handles in its request")
        return self


class FirstDecision(SupervisorDecision):
    """The first decision on a message always states the case."""
    case: Case


class ItemLine(AgentOutput):
    """One order line coming back."""
    item_id: NonEmpty = Field(validation_alias=AliasChoices("item_id", "order_item_id", "id"),
                              description="The order item ID from get_items_for_order, e.g. ITM-0031-2.")
    quantity: int = Field(default=1, ge=1, description="How many units of that line come back.")


# Keys a model may add for its own reasoning; never part of a proposal.
REASONING_KEYS = frozenset({"analysis", "reasoning", "thinking", "thoughts", "thought"})


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

    @model_validator(mode="before")
    @classmethod
    def tidy_model_output(cls, data: Any) -> Any:
        """Undo three habits of models writing JSON: reasoning written into the object (gpt-oss puts its
        analysis channel there in JSON mode on some providers), the proposal nested one level down
        (e.g. under "proposal"), and "items": null for no items."""
        if not isinstance(data, dict):
            return data
        data = {key: value for key, value in data.items() if key not in REASONING_KEYS}
        nested = [value for value in data.values() if isinstance(value, dict) and "action" in value]
        if "action" not in data and len(nested) == 1:
            data = {key: value for key, value in nested[0].items() if key not in REASONING_KEYS}
        return {key: value for key, value in data.items()
                if not (key in {"items", "cited_clauses"} and value is None)}

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
    parts: list[dict] = Field(default_factory=list)   # a multi-part ticket: each specialist's outcome
