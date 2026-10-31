"""One tool contract and dispatcher shared by both supervisor strategies.

Read and ticket-state tools are identical in every architecture; the `policy` tool's arguments
and description come from the plugged-in policy provider (RAG or skills). `backend_write` is for
architectures with no gateway in front of their writes: it runs the shared write gate (rules,
citations and, on the approver route, an independent review) before the executor writes. The
multi-agent specialists are never given it.
"""
from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from support_ops.backend.postgres import (PostgresReader, PostgresWriter, RECORD_KINDS, TicketCategory,
                                          TicketPriority, TicketStatus)
from support_ops.gateway.ticket_state import TicketState
from support_ops.gateway.write_gate import Reviewer, WriteExecutor, WriteGate
from support_ops.helpers.arguments import Amount, NonEmpty, ToolArguments
from support_ops.policy.provider import PolicyProvider
from support_ops.policy.facts import BackendFacts


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
    order_id: NonEmpty | None = Field(default=None, description="Order ID used to load recorded policy facts for a check.")
    item_ids: list[NonEmpty] = Field(default_factory=list)

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
    order_id: NonEmpty | None = Field(default=None, description="Order ID used to load recorded policy facts for a check.")
    item_ids: list[NonEmpty] = Field(default_factory=list)

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
    """Record an audit event or hand off to a human."""
    action: Literal["handoff", "audit"]
    reason: NonEmpty | None = None

    @model_validator(mode="after")
    def required_for_action(self) -> Self:
        if self.action == "handoff" and not self.reason:
            raise ValueError("handoff requires reason")
        return self


class WriteLine(ToolArguments):
    item_id: NonEmpty
    quantity: int = Field(default=1, ge=1)


class BackendWriteArguments(ToolArguments):
    """Write an action to the database: refund (recorded as pending; no money moves), return_request or
    cancel. The recorded facts and the rules are checked first, so the result may be a block, a request
    for review or a handoff instead. A write requiring review goes to an independent reviewer before it
    is executed. Each action is written once per ticket."""
    operation: NonEmpty
    order_id: NonEmpty
    amount: Amount | None = None
    items: list[WriteLine] = Field(default_factory=list, description=(
        "The order lines affected, from get_items_for_order. Required for a return_request."))
    reason: NonEmpty = Field(description="Why, in a short phrase such as damaged_item or order_cancellation.")
    cited_clauses: list[NonEmpty] = Field(default_factory=list, description=(
        "Clause IDs or policy files you retrieved in this ticket that support the action. At least one is "
        "required; citing anything the policy tool did not return is refused."))
    review_reason: NonEmpty | None = Field(default=None, description=(
        "For a write requiring review, explain how the retrieved policy and recorded facts support the "
        "action and amount; an independent reviewer checks it against the evidence. This cannot override "
        "a failed check or human-review requirement."))

    @model_validator(mode="after")
    def required_for_operation(self) -> Self:
        if self.operation == "refund" and self.amount is None:
            raise ValueError("refund write requires amount")
        if self.operation == "return_request" and not self.items:
            raise ValueError("return_request write requires items")
        return self


class UpdateTicketArguments(ToolArguments):
    """Create or update this conversation's support ticket in the database. Call it when the outcome is known."""
    status: TicketStatus
    category: TicketCategory
    priority: TicketPriority = "normal"
    subject: NonEmpty = Field(description="The issue in one sentence.")
    customer_message: NonEmpty = Field(description="The customer's first message; kept from the first save.")
    order_id: NonEmpty | None = Field(default=None, description="An order confirmed with read tools.")
    resolution: NonEmpty | None = Field(default=None, description="What was done, when status is resolved.")


def validate_arguments(models: dict[str, type[ToolArguments]], name: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(name, str) or name not in models:
        raise ValueError(f"unknown tool: {name!r}; available tools: {', '.join(models)}")
    return models[name].model_validate(arguments).model_dump(exclude_none=True)


class SupportTools:
    def __init__(self, reader: PostgresReader, policy: PolicyProvider, ticket_state: TicketState,
                 writer: PostgresWriter | None = None,
                 reviewer: Reviewer | None = None, max_rejections: int = 1):
        self.reader, self.policy, self.ticket_state = reader, policy, ticket_state
        self.writer = writer or PostgresWriter()
        self.backend_facts = BackendFacts(reader, ticket_state.clock)
        # `backend_write` decides with the gate and writes with the executor; they never share a step.
        self.gate = WriteGate(self.backend_facts, ticket_state, reviewer, max_rejections)
        self.executor = WriteExecutor(self.writer, ticket_state)
        self.argument_models: dict[str, type[ToolArguments]] = {
            "read_record": ReadArguments,
            "get_shipments_for_order": OrderArguments,
            "get_shipment_events_for_order": OrderArguments,
            "get_items_for_order": OrderArguments,
            "get_payments_for_order": OrderArguments,
            "get_returns_for_order": OrderArguments,
            "policy": policy.arguments,
            "ticket_state": TicketArguments,
            "backend_write": BackendWriteArguments,
            "update_ticket": UpdateTicketArguments,
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
            "backend_write": BackendWriteArguments.__doc__,
            "update_ticket": UpdateTicketArguments.__doc__,
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
        value = self._dispatch(ticket_id, name, arguments)
        if name not in {"backend_write", "update_ticket", "ticket_state"} and "error" not in value:
            self.gate.record(ticket_id, name, value, self.policy.citable(value) if name == "policy" else set())
        return value

    def _dispatch(self, ticket_id: str, name: str, arguments: Any) -> dict[str, Any]:
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
            if args.get("action") == "check" and args.get("operation") == "delivery" and args.get("order_id"):
                parcels = self.backend_facts.deliveries(args["order_id"])
                checks = [self.policy.run({**args, "facts": parcel}) for parcel in parcels or [{}]]
                decision = next((check for check in checks if check["requires_human"]), checks[0])
                return {**decision, "parcel_checks": checks}
            if args.get("action") == "check":
                args["facts"] = self.backend_facts.read(args["operation"], args["order_id"], args.get("item_ids")) \
                    if args.get("order_id") else {}
            return self.policy.run(args)
        if name == "backend_write":
            return self._backend_write(ticket_id, args)
        if name == "update_ticket":
            row = self.writer.save_ticket(ticket_id=ticket_id, at=self.ticket_state.clock.now(), **args)
            return {"status": "ticket_saved", "ticket_id": row["id"], "ticket_status": row["status"],
                    "customer_id": row["customer_id"], "order_id": row["order_id"]}
        return self.ticket_state.run(ticket_id, args)

    def _backend_write(self, ticket_id: str, args: dict[str, Any]) -> dict[str, Any]:
        """The agent chose to write, so the gate decides and the executor writes: check, then the first
        result of a repeat, then the review if the route needs one, then the write."""
        checked = self.gate.check(ticket_id, args)
        if checked["status"] != "allowed":
            return checked
        decision = checked["policy_check"]
        if (previous := self.executor.replay(ticket_id, args, decision)) is not None:
            return previous   # already reviewed and written: no second review
        if decision["route"] == "approver" and (refused := self.gate.review(ticket_id, args, decision)):
            return refused
        return self.executor.apply(ticket_id, args, decision)
