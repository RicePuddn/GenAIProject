"""One tool contract and dispatcher shared by both supervisor strategies."""
from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from support_ops.backend.postgres import PostgresReader, RECORD_KINDS
from support_ops.gateway.ticket_state import TicketState
from support_ops.policy.skills import SkillsPolicy

NonEmpty = Annotated[str, Field(min_length=1)]
Amount = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReadArguments(ToolArguments):
    """Read a record by its own ID. Shipment lookups require the order lookup tools."""
    kind: Literal[*RECORD_KINDS]
    record_id: NonEmpty


class OrderArguments(ToolArguments):
    order_id: NonEmpty


class PolicyArguments(ToolArguments):
    """list: discover skills; open: read SKILL.md; read: read a discovered clause; check: mock refund limit."""
    action: Literal["list", "open", "read", "check"]
    skill: NonEmpty | None = None
    file: NonEmpty | None = None
    operation: NonEmpty | None = None
    amount: Amount | None = None

    @model_validator(mode="after")
    def required_for_action(self) -> Self:
        if self.action == "open" and not self.skill:
            raise ValueError("open requires skill")
        if self.action == "read" and not (self.skill or self.file):
            raise ValueError("read requires skill or file")
        if self.action == "check" and not self.operation:
            raise ValueError("check requires operation")
        if self.action == "check" and self.operation == "refund" and self.amount is None:
            raise ValueError("refund check requires amount")
        return self


class TicketArguments(ToolArguments):
    """Simulate a guarded write, record an audit event, or hand off to a human. No database mutation."""
    action: Literal["write", "handoff", "audit"]
    operation: NonEmpty | None = None
    record_id: NonEmpty | None = None
    amount: Amount | None = None
    reason: NonEmpty | None = None

    @model_validator(mode="after")
    def required_for_action(self) -> Self:
        if self.action == "write" and not (self.operation and self.record_id):
            raise ValueError("write requires operation and record_id")
        if self.action == "write" and self.operation == "refund" and self.amount is None:
            raise ValueError("refund write requires amount")
        if self.action == "handoff" and not self.reason:
            raise ValueError("handoff requires reason")
        return self


ARGUMENT_MODELS = {
    "read_record": ReadArguments,
    "get_shipments_for_order": OrderArguments,
    "get_shipment_events_for_order": OrderArguments,
    "policy": PolicyArguments,
    "ticket_state": TicketArguments,
}
DESCRIPTIONS = {
    "read_record": ReadArguments.__doc__,
    "get_shipments_for_order": "Get all shipments for an order ID, not a shipment ID.",
    "get_shipment_events_for_order": "Get chronological carrier tracking events for an order ID.",
    "policy": PolicyArguments.__doc__,
    "ticket_state": TicketArguments.__doc__,
}
TOOLS = [
    {"type": "function", "function": {
        "name": name, "description": DESCRIPTIONS[name], "parameters": model.model_json_schema(),
    }}
    for name, model in ARGUMENT_MODELS.items()
]


def validate_arguments(name: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(name, str) or name not in ARGUMENT_MODELS:
        raise ValueError(f"unknown tool: {name!r}; available tools: {', '.join(ARGUMENT_MODELS)}")
    return ARGUMENT_MODELS[name].model_validate(arguments).model_dump(exclude_none=True)


class SupportTools:
    def __init__(self, reader: PostgresReader, skills: SkillsPolicy, ticket_state: TicketState):
        self.reader, self.skills, self.ticket_state = reader, skills, ticket_state

    def execute(self, ticket_id: str, name: str, arguments: Any) -> dict[str, Any]:
        args = validate_arguments(name, arguments)
        if name == "read_record":
            return self.reader.read(**args)
        if name == "get_shipments_for_order":
            return {"shipments": self.reader.shipments_for_order(**args)}
        if name == "get_shipment_events_for_order":
            return {"events": self.reader.shipment_events_for_order(**args)}
        if name == "policy":
            return self.skills.run(args)
        return self.ticket_state.run(ticket_id, args)
