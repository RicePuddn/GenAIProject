"""The one tool registry. Every agent in every setup gets its data tools from here: the single agent all of
them, each specialist a subset, the graph setups' gather stage only the read and policy tools.

`Toolset.call` is the one way a tool runs: it counts against the ticket's budget, applies the retry
policy, logs the call and adds it to the turn's evidence. A tool returns a result for the agent; a
mistake the agent can fix (bad arguments, a missing file) comes back as {"error": ...}.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Callable, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from support_ops.shared.budget import BudgetExceeded
from support_ops.shared.cancellation import WorkflowCancelled
from support_ops.shared.db import RECORD_KINDS

if TYPE_CHECKING:
    from support_ops.shared.context import TurnContext
    from support_ops.shared.db import PostgresReader
    from support_ops.shared.gates.write_gate import WriteExecutor, WriteGate
    from support_ops.shared.policy.provider import PolicyProvider

NonEmpty = Annotated[str, Field(min_length=1)]
Amount = Annotated[float, Field(gt=0, allow_inf_nan=False)]

READ_TOOLS = ("read_record", "get_orders_for_customer", "get_shipments_for_order", "get_shipment_events_for_order",
              "get_items_for_order", "get_payments_for_order", "get_returns_for_order")
GATHER_TOOLS = (*READ_TOOLS, "policy")            # what a graph setup's agent may use
WRITE_TOOLS = ("backend_write", "request_review")  # agentic setups only
ALL_TOOLS = (*GATHER_TOOLS, *WRITE_TOOLS)

# Every write is checked by backend_write itself, so `check` covers only what no write covers: the delivery
# rules for an order's parcels.
CHECK_DESCRIPTION = ("check: the delivery rules for an order's parcels (lost, late or damaged in transit), "
                     "with operation delivery and the order_id. Writes need no check: backend_write applies "
                     "the rules itself.")
CHECK_NOTE = ("rule_sources name the sections this rule comes from, without their text: they cannot be cited. "
              "Cite clause IDs or files from a policy search or read.")


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReadArguments(ToolArguments):
    """Read a record by its own ID. For an order's items, payments, returns or shipments, use the order lookup tools."""
    kind: Literal[*RECORD_KINDS]
    record_id: NonEmpty


class CustomerOrdersArguments(ToolArguments):
    """List the signed-in customer's latest orders, newest first: order ID, date, status, total and product
    names. Use it when the customer has not given an order number, to find the order they mean."""


class OrderArguments(ToolArguments):
    order_id: NonEmpty


class WriteLine(ToolArguments):
    item_id: NonEmpty
    quantity: int = Field(default=1, ge=1)


class BackendWriteArguments(ToolArguments):
    """Write an action to the database: refund (recorded as pending; no money moves), return_request or
    cancel. The recorded facts and the rules are checked first, so the result may be a block, a request
    for review or a handoff instead. A write requiring review is refused (review_required) until
    request_review approves that exact write: the same operation, order, amount and items. Each action is
    written once per ticket."""
    operation: NonEmpty
    order_id: NonEmpty
    amount: Amount | None = None
    items: list[WriteLine] = Field(default_factory=list, description=(
        "The order lines affected, from get_items_for_order. Required for a return_request."))
    reason: NonEmpty = Field(description="Why, in a short phrase such as damaged_item or order_cancellation.")
    cited_clauses: list[NonEmpty] = Field(default_factory=list, description=(
        "Clause IDs or policy files a policy search or read returned in this ticket that support the action. "
        "At least one is required; citing anything else is refused."))

    @model_validator(mode="after")
    def required_for_operation(self) -> Self:
        if self.operation == "refund" and self.amount is None:
            raise ValueError("refund write requires amount")
        if self.operation == "return_request" and not self.items:
            raise ValueError("return_request write requires items")
        return self


class RequestReviewArguments(BackendWriteArguments):
    """Ask the independent reviewer to approve a write before making it. Give exactly the write you will
    make, with review_reason explaining how the cited policy and recorded facts support it. A write that
    needs a review is refused until approved; an approval covers only the same operation, order, amount
    and items."""
    review_reason: NonEmpty = Field(description=(
        "How the retrieved policy and recorded facts support the action and amount; the reviewer checks it "
        "against the evidence. It cannot override a failed check or a human requirement."))


class FinishArguments(ToolArguments):
    """Finish: say how this ends, once the work is done. This writes nothing."""
    outcome: Literal["resolved", "awaiting_customer", "awaiting_human"] = Field(description=(
        "resolved: answered or done; awaiting_customer: only the customer can supply a missing fact; "
        "awaiting_human: a person must decide."))
    note: NonEmpty = Field(description=(
        "resolved: what was done or found; awaiting_customer: the question to ask; awaiting_human: why."))


def validate_arguments(model: type[ToolArguments], arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):   # models often send null for an optional field: treat it as not given
        arguments = {key: value for key, value in arguments.items() if value is not None}
    return model.model_validate(arguments).model_dump(exclude_none=True)


@dataclass(frozen=True)
class Tool:
    name: str
    arguments: type[ToolArguments]
    description: str
    run: Callable[[dict[str, Any]], dict[str, Any]]   # validated arguments -> result
    data: bool = True   # a data tool counts as a tool call and goes into the evidence; a control tool does not

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": " ".join(self.description.split()),
                                                 "parameters": _untitled(self.arguments.model_json_schema())}}


def _untitled(schema: Any) -> Any:
    """A JSON schema without pydantic's generated titles ("Order Id"), which only cost tokens. A property
    that is itself named title would be kept: only a title alongside a schema keyword is dropped."""
    if isinstance(schema, dict):
        return {key: _untitled(value) for key, value in schema.items()
                if not (key == "title" and isinstance(value, str) and ("type" in schema or "properties" in schema
                                                                        or "anyOf" in schema or "$ref" in schema))}
    if isinstance(schema, list):
        return [_untitled(item) for item in schema]
    return schema


def finish_tool() -> Tool:
    return Tool("finish", FinishArguments, FinishArguments.__doc__, lambda args: {"status": "finished"}, data=False)


class Toolset:
    """The tools one agent holds. `terminal` names the tool that ends its loop (finish), if it has one."""

    def __init__(self, tools: Sequence[Tool], terminal: str | None = None):
        self.tools = {tool.name: tool for tool in tools}
        if terminal is not None and terminal not in self.tools:
            raise ValueError(f"terminal tool {terminal!r} is not in the toolset")
        self.terminal = terminal

    @property
    def names(self) -> list[str]:
        return list(self.tools)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self.tools.values()]

    def validate(self, name: Any, arguments: Any) -> dict[str, Any]:
        if not isinstance(name, str) or name not in self.tools:
            raise ValueError(f"unknown tool: {name!r}; available tools: {', '.join(self.tools)}")
        return validate_arguments(self.tools[name].arguments, arguments)

    def call(self, ctx: TurnContext, agent: str, call_id: str, name: str, arguments: Any) -> dict[str, Any]:
        """Run one tool call for `agent`. Errors the agent can fix are its result; a failure it cannot fix
        (e.g. a database timeout) is re-run `retries.tool_errors` times first."""
        tool = self.tools.get(name) if isinstance(name, str) else None
        data = tool is None or tool.data
        # A control tool (finish, ask_specialist) is traced apart, so tool counts are data tools in every setup.
        ctx.log("tool_call_requested" if data else "control_call", call_id=call_id, tool=name, input=arguments,
                agent=agent)
        if data:
            if tool is not None:
                ctx.budget.tool_call()
            ctx.log("tool_start", call_id=call_id, tool=name, input=arguments, agent=agent)
        started = time.perf_counter()
        attempts = ctx.retries.tool_errors + 1
        args: Any = arguments
        for attempt in range(1, attempts + 1):
            try:
                args = self.validate(name, arguments)
                value = tool.run(args)
            except (BudgetExceeded, WorkflowCancelled):
                raise
            except (ValueError, FileNotFoundError) as exc:   # the same arguments would fail again
                value = {"error": str(exc)}
            except Exception as exc:
                value = {"error": str(exc)}
                if attempt < attempts:
                    ctx.log("tool_retry", call_id=call_id, tool=name, attempt=attempt, error=str(exc))
                    time.sleep(ctx.retries.backoff_s * attempt)
                    continue
            break
        ctx.log("tool_end" if data else "control_result", call_id=call_id, tool=name, agent=agent,
                elapsed_s=round(time.perf_counter() - started, 6), output=value)
        if data:
            ctx.evidence.append({"agent": agent, "tool": name, "arguments": args, "result": value})
        return value


class ToolRegistry:
    """The data tools for one ticket's turn, on its account-scoped reader and its write gate."""

    def __init__(self, reader: PostgresReader, policy: PolicyProvider, gate: WriteGate, executor: WriteExecutor):
        self.reader, self.policy, self.gate, self.executor = reader, policy, gate, executor
        lookups = {
            "get_shipments_for_order": ("Get all shipments for an order ID, not a shipment ID.",
                                        lambda order_id: {"shipments": reader.shipments_for_order(order_id)}),
            "get_shipment_events_for_order": ("Get chronological carrier tracking events for an order ID.",
                                              lambda order_id: {"events": reader.shipment_events_for_order(order_id)}),
            "get_items_for_order": ("Get the items in an order ID: product name, SKU, category, quantity, unit "
                                    "price and fulfilment status.",
                                    lambda order_id: {"items": reader.items_for_order(order_id)}),
            "get_payments_for_order": ("Get the payments for an order ID: method, status and amount.",
                                       lambda order_id: {"payments": reader.payments_for_order(order_id)}),
            "get_returns_for_order": ("Get existing returns and refunds for an order ID, with their status.",
                                      reader.returns_for_order),
        }
        tools = [
            Tool("read_record", ReadArguments, ReadArguments.__doc__, lambda args: reader.read(**args)),
            Tool("get_orders_for_customer", CustomerOrdersArguments, CustomerOrdersArguments.__doc__,
                 lambda args: self._own_orders()),
            *(Tool(name, OrderArguments, description, lambda args, run=run: run(args["order_id"]))
              for name, (description, run) in lookups.items()),
            Tool("policy", policy.arguments, policy.description, lambda args: self._policy(args)),
            Tool("backend_write", BackendWriteArguments, BackendWriteArguments.__doc__, self._backend_write),
            Tool("request_review", RequestReviewArguments, RequestReviewArguments.__doc__, gate.request_review),
        ]
        self.tools = {tool.name: tool for tool in tools}
        assert tuple(self.tools) == ALL_TOOLS

    def toolset(self, names: Sequence[str], *, extra: Sequence[Tool] = (), terminal: str | None = None,
                policy: PolicyProvider | None = None) -> Toolset:
        """The named tools, plus `extra`. `policy` replaces the policy tool's provider, e.g. with a
        specialist's scoped view of it."""
        if unknown := sorted(set(names) - set(self.tools)):
            raise ValueError(f"unknown tools: {unknown}")
        tools = {name: self.tools[name] for name in names}
        if policy is not None and "policy" in tools:
            tools["policy"] = Tool("policy", policy.arguments, policy.description,
                                   lambda args: self._policy(args, policy))
        return Toolset([*tools.values(), *extra], terminal)

    def _own_orders(self) -> dict[str, Any]:
        # Only an account-scoped reader can list orders, and only its own customer's.
        if not hasattr(self.reader, "own_orders"):
            raise ValueError("no customer is signed in, so their orders cannot be listed: "
                             "ask the customer for the order number")
        return {"orders": self.reader.own_orders()}

    def _policy(self, args: dict[str, Any], policy: PolicyProvider | None = None) -> dict[str, Any]:
        policy = policy or self.policy
        if args["action"] != "check":
            return policy.run(args)
        # The argument models allow only the delivery check: each parcel is checked on its own.
        parcels = self.gate.facts.deliveries(args["order_id"])
        checks = [policy.run({**args, "facts": parcel}) for parcel in parcels or [{}]]
        decision = next((check for check in checks if check["requires_human"]), checks[0])
        return {**decision, "parcel_checks": checks, "note": CHECK_NOTE}

    def _backend_write(self, args: dict[str, Any]) -> dict[str, Any]:
        """The agent chose to write: a repeat gets its first result; otherwise the gate checks it, a write
        on the approver route needs request_review's approval, then the executor writes."""
        if (previous := self.executor.replay(args)) is not None:
            return previous
        checked = self.gate.check(args)
        if checked["status"] != "allowed":
            return checked
        decision = checked["policy_check"]
        if decision["route"] == "approver" and not self.gate.is_approved(args):
            return {"status": "review_required", "policy_check": decision,
                    "reason": "This write needs an independent review first: call request_review with the same "
                              "operation, order, amount and items and a review_reason, then write it once approved."}
        return self.executor.apply(args, decision)
