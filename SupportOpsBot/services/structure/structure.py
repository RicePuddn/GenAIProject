"""Strict contracts shared by tools, agents, and graph-controlled workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from pydantic import (AliasChoices, BaseModel, ConfigDict, Field, RootModel,
                      model_validator)

NonEmpty = Annotated[str, Field(min_length=1)]
Amount = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ToolArguments(StrictModel):
    """Base class for all model supplied tool input."""


class GetCustomerOrdersInput(ToolArguments):
    """List only the orders belonging to the signed in customer."""


class OrderIdInput(ToolArguments):
    order_id: NonEmpty


class GetOrderInput(ToolArguments):
    order_id: NonEmpty


class GetOrderItemsInput(OrderIdInput):
    pass


class GetShipmentsInput(OrderIdInput):
    pass


class GetTrackingEventsInput(OrderIdInput):
    pass


class GetPaymentsInput(OrderIdInput):
    pass


class GetReturnsAndRefundsInput(OrderIdInput):
    pass


class WriteLine(ToolArguments):
    item_id: NonEmpty = Field(validation_alias=AliasChoices("item_id", "order_item_id", "id"))
    quantity: int = Field(default=1, ge=1)


class WriteFields(ToolArguments):
    order_id: NonEmpty
    reason: NonEmpty
    cited_clauses: list[NonEmpty] = Field(min_length=1)
    policy_exception: bool = False
    route: Literal["approver", "human"] | None = None


class CreateReturnRequestInput(WriteFields):
    items: list[WriteLine] = Field(min_length=1)


class RequestRefundInput(WriteFields):
    amount: Amount


class CancelOrderInput(WriteFields):
    pass


class BackendWriteArguments(StrictModel):
    """Internal shared write contract used by the write gate and graph stages."""
    operation: Literal["refund", "return_request", "cancel"]
    order_id: NonEmpty
    amount: Amount | None = None
    items: list[WriteLine] = Field(default_factory=list)
    reason: NonEmpty
    cited_clauses: list[NonEmpty] = Field(default_factory=list)
    policy_exception: bool = False
    route: Literal["approver", "human"] | None = None

    @model_validator(mode="after")
    def required_for_operation(self):
        if self.operation == "refund" and self.amount is None:
            raise ValueError("refund write requires amount")
        if self.operation == "return_request" and not self.items:
            raise ValueError("return_request write requires items")
        return self


class RequestReviewArguments(BackendWriteArguments):
    review_reason: NonEmpty


class FinishArguments(ToolArguments):
    """End an agentic run after evidence gathering and any permitted actions."""
    outcome: Literal["resolved", "awaiting_customer", "awaiting_human"]
    note: NonEmpty


class FinishOutput(StrictModel):
    status: Literal["finished"]


class AskSpecialistArguments(ToolArguments):
    """Delegate one focused part of the customer's support request to a specialist."""
    specialist: Literal["orders_shipping", "returns_refunds"]
    request: NonEmpty


class AskSpecialistOutput(StrictModel):
    specialist: Literal["orders_shipping", "returns_refunds"]
    outcome: Literal["resolved", "awaiting_customer", "awaiting_human"]
    note: str | None = None
    writes: list[dict[str, Any]] = Field(default_factory=list)


class ToolError(StrictModel):
    """Typed error envelope returned by a tool handler or argument validator."""
    error: NonEmpty
    code: Literal["invalid_arguments", "tool_error", "invalid_handler_output", "repeated_non_retryable_call"]
    retryable: bool
    field_errors: list[str] = Field(default_factory=list)


class AnyObjectOutput(RootModel[dict[str, Any]]):
    """A database record or policy document whose columns depend on the selected table/provider."""


class CustomerOrdersOutput(StrictModel):
    orders: list[dict[str, Any]]


class ItemsOutput(StrictModel):
    items: list[dict[str, Any]]


class ShipmentsOutput(StrictModel):
    shipments: list[dict[str, Any]]


class TrackingEventsOutput(StrictModel):
    events: list[dict[str, Any]]


class PaymentsOutput(StrictModel):
    payments: list[dict[str, Any]]


class ReturnsAndRefundsOutput(StrictModel):
    returns: list[dict[str, Any]]
    refunds: list[dict[str, Any]]


class ActionOutput(StrictModel):
    status: NonEmpty
    reason: str | None = None
    operation: str | None = None
    order_id: str | None = None
    amount: Any | None = None
    write_id: str | None = None
    replayed: bool | None = None
    policy_check: dict[str, Any] | None = None
    existing_requests: list[dict[str, Any]] | None = None
    review: dict[str, Any] | None = None
    error: str | None = None


class AgentOutput(BaseModel):
    # LLM structured outputs permit numeric coercion; unknown keys remain rejected.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProposedLine(AgentOutput):
    item_id: NonEmpty = Field(validation_alias=AliasChoices("item_id", "order_item_id", "id"),
                              description="The order item ID returned by get_order_items.")
    quantity: int = Field(default=1, ge=1)


class ProposedWrite(AgentOutput):
    operation: Literal["refund", "return_request", "cancel"]
    order_id: NonEmpty
    amount: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    items: list[ProposedLine] = Field(default_factory=list)
    reason: NonEmpty
    cited_clauses: list[NonEmpty] = Field(default_factory=list)
    policy_exception: bool = False
    route: Literal["approver", "human"] | None = None


class Proposal(AgentOutput):
    outcome: Literal["resolved", "awaiting_customer", "awaiting_human"]
    note: NonEmpty
    writes: list[ProposedWrite] = Field(default_factory=list)
    review_reason: NonEmpty | None = None

    @model_validator(mode="before")
    @classmethod
    def tidy_model_output(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        reasoning = {"analysis", "reasoning", "thinking", "thoughts", "thought"}
        data = {key: value for key, value in data.items() if key not in reasoning}
        if data.get("proposed_line") == []:
            data.pop("proposed_line")
        nested = [value for value in data.values() if isinstance(value, dict) and "outcome" in value]
        if "outcome" not in data and len(nested) == 1:
            data = {key: value for key, value in nested[0].items() if key not in reasoning}
        return {key: value for key, value in data.items() if not (key == "writes" and value is None)}


@dataclass
class PartResult:
    status: Literal["resolved", "awaiting_customer", "awaiting_human"]
    note: str | None
    writes: list[dict[str, Any]] = field(default_factory=list)


class PartState(TypedDict, total=False):
    feedback: str | None
    proposal: Proposal | None
    pending: list[dict[str, Any]]
    decision: dict[str, Any] | None
    attempt: dict[str, Any] | None
    next: str

