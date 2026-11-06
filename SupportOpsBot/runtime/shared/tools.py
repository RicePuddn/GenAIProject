"""The one tool registry. Every agent in every setup gets its data tools from here: the single agent all of
them, each specialist a subset, the graph setups' gather stage only the read and policy tools.

`Toolset.call` is the one way a tool runs: it counts against the ticket's budget, applies the retry
policy, logs the call and adds it to the turn's evidence. A tool returns a result for the agent; a
mistake the agent can fix (bad arguments, a missing file) comes back as {"error": ...}.
"""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ValidationError

from SupportOpsBot.services.structure.structure import (
    ToolArguments, ToolError,
)
from SupportOpsBot.services.tools.tool import Tool

from SupportOpsBot.runtime.shared.budget import BudgetExceeded
from SupportOpsBot.runtime.shared.cancellation import WorkflowCancelled
from SupportOpsBot.runtime.shared.llm import ProviderError
from SupportOpsBot.runtime.shared.tool_catalog import ALL_TOOLS
from SupportOpsBot.runtime.shared.gates.write_gate import WriteCoordinator

if TYPE_CHECKING:
    from SupportOpsBot.runtime.shared.context import TurnContext
    from SupportOpsBot.runtime.shared.db import PostgresReader
    from SupportOpsBot.runtime.shared.gates.write_gate import WriteExecutor, WriteGate
    from SupportOpsBot.runtime.shared.policy.provider import PolicyProvider

def run_with_retries(ctx: TurnContext, call_id: str, name: str, arguments: Any, run: Callable[[], dict[str, Any]],
                     attempts: int | None = None) -> dict[str, Any]:
    """Run a tool's work under the one retry policy. An error the agent can fix is its result, at once; a
    failure it cannot fix (e.g. a database timeout) is re-run `retries.tool_errors` times first. One that
    still fails stays in `ctx.failed_calls` until the same call works, so the turn cannot end resolved."""
    attempts = attempts or ctx.retries.tool_errors + 1
    key = f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"
    for attempt in range(1, attempts + 1):
        try:
            value = run()
            ctx.failed_calls.pop(key, None)
            return value
        except (BudgetExceeded, WorkflowCancelled, ProviderError):
            raise
        except (ValueError, FileNotFoundError) as exc:   # the same arguments would fail again
            return {"error": str(exc)}
        except Exception as exc:
            if attempt == attempts:
                ctx.failed_calls[key] = str(exc)
                return {"error": str(exc)}
            ctx.log("tool_retry", call_id=call_id, tool=name, attempt=attempt, error=str(exc))
            time.sleep(ctx.retries.backoff_s * attempt)
    raise AssertionError("unreachable")


def validate_arguments(model: type[ToolArguments], arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):   # models often send null for an optional field: treat it as not given
        arguments = {key: value for key, value in arguments.items() if value is not None}
    return model.model_validate(arguments).model_dump(exclude_none=True)


def begin_data_call(ctx: TurnContext, agent: str, name: str, arguments: Any, *, call_id: str | None = None,
                    node: str | None = None, count_budget: bool = True) -> dict[str, Any]:
    call_id = call_id or ctx.next_call_id()
    if node:
        ctx.log("node_start", node=node, agent=agent)
    ctx.log("tool_call_requested", call_id=call_id, tool=name, input=arguments, agent=agent)
    if count_budget:
        ctx.budget.tool_call()
    ctx.log("tool_start", call_id=call_id, tool=name, input=arguments, agent=agent)
    return {"call_id": call_id, "started": time.perf_counter()}


def end_data_call(ctx: TurnContext, agent: str, name: str, arguments: Any, value: dict[str, Any],
                  attempt: dict[str, Any]) -> None:
    ctx.log("tool_end", call_id=attempt["call_id"], tool=name, agent=agent,
            elapsed_s=round(time.perf_counter() - attempt["started"], 6), output=value)
    ctx.evidence.append({"agent": agent, "tool": name, "arguments": arguments, "result": value})


def finish_tool() -> Tool:
    from SupportOpsBot.services.tools.finish import create_tool
    return create_tool()


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

    def resolve_terminal_call(self, name: Any, arguments: Any) -> tuple[Any, Any]:
        """Repair an unknown tool name only when its arguments exactly validate as the terminal control
        tool. This handles a model that emits a generic name (for example, `json`) with a valid finish
        payload, without aliasing names or redirecting malformed calls to data or write tools."""
        if isinstance(name, str) and name in self.tools:
            return name, arguments
        if self.terminal is None or self.tools[self.terminal].data:
            return name, arguments
        if not isinstance(arguments, dict):
            return name, arguments
        schema = self.tools[self.terminal].arguments.model_json_schema()
        properties = set(schema.get("properties", {}))
        required = set(schema.get("required", []))
        if not required.issubset(arguments) or not set(arguments).issubset(properties):
            return name, arguments
        try:
            validated = self.validate(self.terminal, arguments)
        except (TypeError, ValueError, ValidationError):
            return name, arguments
        return self.terminal, validated

    def call(self, ctx: TurnContext, agent: str, call_id: str, name: str, arguments: Any) -> dict[str, Any]:
        """Run one tool call for `agent`. A data tool runs under the retry policy; a control tool is not
        re-run, because asking a specialist again would repeat its whole part."""
        tool = self.tools.get(name) if isinstance(name, str) else None
        data = tool is None or tool.data
        # A control tool (finish, ask_specialist) is traced apart, so tool counts are data tools in every setup.
        if data:
            attempt = begin_data_call(ctx, agent, name, arguments, call_id=call_id, count_budget=tool is not None)
        else:
            ctx.log("control_call", call_id=call_id, tool=name, input=arguments, agent=agent)
        started = time.perf_counter()
        args: Any = arguments
        value: dict[str, Any]
        try:
            args = self.validate(name, arguments)
        except (TypeError, ValueError, ValidationError) as exc:
            failures = ctx.invalid_argument_calls.get(agent, 0) + 1
            ctx.invalid_argument_calls[agent] = failures
            fields = _argument_errors(exc)
            fatal = failures > 1
            message = ("Invalid arguments. Correct these fields and call the tool once more: "
                       + "; ".join(fields))
            if fatal:
                message = ("A second invalid tool call ended this agent run. "
                           + "; ".join(fields))
                ctx.stop(f"{agent} made a second invalid tool call")
            value = ToolError(error=message, code="invalid_arguments", retryable=not fatal,
                              field_errors=fields).model_dump(mode="json")
        else:
            call_key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            if call_key in ctx.non_retryable_outputs:
                message = "The same tool call previously returned an invalid handler result and cannot be repeated."
                ctx.stop(f"{name} repeated a non-retryable failing call")
                value = ToolError(error=message, code="repeated_non_retryable_call", retryable=False).model_dump(
                    mode="json")
            else:
                value = run_with_retries(ctx, call_id, name, args, lambda: tool.run(args), None if data else 1)
                if isinstance(value, dict) and "error" in value:
                    value = ToolError(error=str(value["error"]), code="tool_error", retryable=True).model_dump(
                        mode="json")
                elif tool is not None and tool.output_model is not None:
                    try:
                        value = tool.output_model.model_validate(value).model_dump(mode="json")
                    except (TypeError, ValueError, ValidationError) as exc:
                        message = f"Tool handler returned invalid output: {exc}"
                        ctx.non_retryable_outputs[call_key] = message
                        value = ToolError(error=message, code="invalid_handler_output", retryable=False).model_dump(
                            mode="json")
        if data:
            end_data_call(ctx, agent, name, args, value, attempt)
        else:
            ctx.log("control_result", call_id=call_id, tool=name, agent=agent,
                    elapsed_s=round(time.perf_counter() - started, 6), output=value)
        return value


def _argument_errors(exc: Exception) -> list[str]:
    if isinstance(exc, ValidationError):
        return [f"{'.'.join(map(str, item['loc'])) or 'arguments'}: {item['msg']}"
                for item in exc.errors(include_url=False)]
    return [str(exc)]


class ToolRegistry:
    """The data tools for one ticket's turn, on its account-scoped reader and its write gate."""

    def __init__(self, reader: PostgresReader, policy: PolicyProvider, gate: WriteGate, executor: WriteExecutor):
        self.reader, self.policy, self.gate, self.executor = reader, policy, gate, executor
        self.writes = WriteCoordinator(gate, executor)
        from SupportOpsBot.services.tools import read, write

        tools = [
            read.get_customer_orders.create_tool(self),
            read.get_order.create_tool(self),
            read.get_order_items.create_tool(self),
            read.get_shipments.create_tool(self),
            read.get_tracking_events.create_tool(self),
            read.get_payments.create_tool(self),
            read.get_returns_and_refunds.create_tool(self),
            read.policy.create_tool(self),
            write.create_return_request.create_tool(self),
            write.request_refund.create_tool(self),
            write.cancel_order.create_tool(self),
            write.request_review.create_tool(self),
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
            from SupportOpsBot.services.tools.read.policy import create_tool
            tools["policy"] = create_tool(self, policy)
        return Toolset([*tools.values(), *extra], terminal)
