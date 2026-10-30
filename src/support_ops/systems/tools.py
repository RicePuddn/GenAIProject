"""One tool contract and dispatcher shared by both supervisor strategies.

Read and ticket-state tools are identical in every architecture; the `policy` tool's arguments
and description come from the plugged-in policy provider (RAG or skills).
"""
from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from support_ops.backend.postgres import PostgresReader, RECORD_KINDS
from support_ops.gateway.ticket_state import TicketState
from support_ops.helpers.arguments import Amount, NonEmpty, ToolArguments
from support_ops.policy.provider import PolicyProvider


RULE_FACTS = (
    "Use recorded facts, never assumed customer claims. "
    "refund: payment_status, currency, captured_amount, refunded_amount, pending_refund_amount, "
    "approved_refund_amount, original_payment_method, request_accepted, inspection_or_waiver. "
    "return_request: delivered_at, requested_at, reason (change_of_mind, damaged, faulty, wrong_item, missing_item), "
    "evidence_status; change_of_mind also needs final_sale, unused, original_packaging, tags_complete, accessories_complete. "
    "return_dispatch: approved_at, handed_over_at, carrier_accepted; extension_until and extension_approved if applicable. "
    "cancel: order_status, label_created, packing_started, any_quantity_fulfilled, carrier_collected. "
    "delivery: delivery_status, estimated_delivery, as_of. "
    "Dates are ISO dates/timestamps; condition flags are booleans. Missing facts require human review."
)


class ReadArguments(ToolArguments):
    """Read a record by its own ID. For an order's items, payments, returns or shipments, use the order lookup tools."""
    kind: Literal[*RECORD_KINDS]
    record_id: NonEmpty


class OrderArguments(ToolArguments):
    order_id: NonEmpty


class PolicyArguments(ToolArguments):
    """list: discover skills; open: read SKILL.md; read: read a discovered clause; check: deterministic refund, return_request, return_dispatch, cancel or delivery checks using recorded facts."""
    action: Literal["list", "open", "read", "check"]
    skill: NonEmpty | None = None
    file: NonEmpty | None = None
    operation: NonEmpty | None = None
    amount: Amount | None = None
    facts: dict[str, Any] | None = Field(default=None, description=RULE_FACTS)

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


class RagPolicyArguments(ToolArguments):
    """search: retrieve the policy clauses most relevant to a query, with their exceptions; check: deterministic refund, return_request, return_dispatch, cancel or delivery checks using recorded facts."""
    action: Literal["search", "check"]
    query: NonEmpty | None = None
    operation: NonEmpty | None = None
    amount: Amount | None = None
    facts: dict[str, Any] | None = Field(default=None, description=RULE_FACTS)

    @model_validator(mode="after")
    def required_for_action(self) -> Self:
        if self.action == "search" and not self.query:
            raise ValueError("search requires query")
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
    facts: dict[str, Any] | None = Field(default=None, description=RULE_FACTS)
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


def validate_arguments(models: dict[str, type[ToolArguments]], name: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(name, str) or name not in models:
        raise ValueError(f"unknown tool: {name!r}; available tools: {', '.join(models)}")
    return models[name].model_validate(arguments).model_dump(exclude_none=True)


class SupportTools:
    def __init__(self, reader: PostgresReader, policy: PolicyProvider, ticket_state: TicketState):
        self.reader, self.policy, self.ticket_state = reader, policy, ticket_state
        self.argument_models: dict[str, type[ToolArguments]] = {
            "read_record": ReadArguments,
            "get_shipments_for_order": OrderArguments,
            "get_shipment_events_for_order": OrderArguments,
            "get_items_for_order": OrderArguments,
            "get_payments_for_order": OrderArguments,
            "get_returns_for_order": OrderArguments,
            "policy": policy.arguments,
            "ticket_state": TicketArguments,
        }
        descriptions = {
            "read_record": ReadArguments.__doc__,
            "get_shipments_for_order": "Get all shipments for an order ID, not a shipment ID.",
            "get_shipment_events_for_order": "Get chronological carrier tracking events for an order ID.",
            "get_items_for_order": "Get the items in an order ID: product name, SKU, category, quantity, "
                                   "unit price and fulfilment status.",
            "get_payments_for_order": "Get the payments for an order ID: method, status and amount.",
            "get_returns_for_order": "Get existing returns and refunds for an order ID, with their status.",
            "policy": policy.description,
            "ticket_state": TicketArguments.__doc__,
        }
        self.schemas = [
            {"type": "function", "function": {
                "name": name, "description": descriptions[name], "parameters": model.model_json_schema(),
            }}
            for name, model in self.argument_models.items()
        ]

    def validate(self, name: str, arguments: Any) -> dict[str, Any]:
        return validate_arguments(self.argument_models, name, arguments)

    def execute(self, ticket_id: str, name: str, arguments: Any) -> dict[str, Any]:
        args = self.validate(name, arguments)
        if name == "read_record":
            return self.reader.read(**args)
        if name == "get_shipments_for_order":
            return {"shipments": self.reader.shipments_for_order(**args)}
        if name == "get_shipment_events_for_order":
            return {"events": self.reader.shipment_events_for_order(**args)}
        if name == "get_items_for_order":
            return {"items": self.reader.items_for_order(**args)}
        if name == "get_payments_for_order":
            return {"payments": self.reader.payments_for_order(**args)}
        if name == "get_returns_for_order":
            return self.reader.returns_for_order(**args)
        if name == "policy":
            return self.policy.run(args)
        return self.ticket_state.run(ticket_id, args)
