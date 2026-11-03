"""One turn of one ticket, as every agent and gate in it sees it.

`TurnContext` carries the LLM, the trace, the clock, the budget and the retry policy, and collects the
evidence: every data tool call any agent makes, in order. A gate that hands the ticket to a person calls
`stop`, and every agent loop ends at its next step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from SupportOpsBot.runtime.shared.budget import Budget, Retries
from SupportOpsBot.runtime.shared.cancellation import CancellationToken
from SupportOpsBot.runtime.shared.clock import Clock
from SupportOpsBot.runtime.shared.llm import LLM, ChatResult
from SupportOpsBot.runtime.shared.tracing import Tracer
from SupportOpsBot.runtime.shared.tool_catalog import WRITE_ACTIONS


@dataclass
class TurnContext:
    cfg: dict[str, Any]
    ticket_id: str
    llm: LLM
    clock: Clock
    tracer: Tracer
    budget: Budget
    retries: Retries
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    # Every data tool call in this turn: {"agent", "tool", "arguments", "result"}.
    evidence: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    # Tool calls that failed for a reason no agent can fix and never succeeded after: {call key: error}.
    failed_calls: dict[str, str] = field(default_factory=dict)
    invalid_argument_calls: dict[str, int] = field(default_factory=dict)
    non_retryable_outputs: dict[str, str] = field(default_factory=dict)
    _call_count: int = 0

    def log(self, kind: str, **payload: Any) -> None:
        """Add an event to the trace. Each one is also a point where a cancelled turn stops."""
        self.cancellation.check()
        self.tracer.log(kind, simulated_at=self.clock.now().isoformat(), **payload)

    def chat(self, role: str, messages: list[dict[str, Any]], *, tools: list[dict] | None = None,
             response_schema: dict[str, Any] | None = None) -> ChatResult:
        self.log("llm_request", role=role)
        previous_llm_calls = self.tracer.llm_calls
        result = self.llm.chat(role, messages, tools=tools, tracer=self.tracer,
                               response_schema=response_schema)
        latency = None
        if self.tracer.llm_calls > previous_llm_calls:
            latency = self.tracer.of_kind("llm_call")[-1]["latency_s"]
        self.log("llm_response", role=role, tool_calls=[call.name for call in result.tool_calls],
                 **({"latency_s": latency} if latency is not None else {}),
                 **({"reasoning": result.reasoning} if result.reasoning else {}))
        return result

    def next_call_id(self) -> str:
        self._call_count += 1
        return f"call-{self._call_count}"

    def stop(self, reason: str) -> None:
        """A gate handed the ticket to a person: the turn ends, whichever agent is working."""
        if self.stop_reason is None:
            self.stop_reason = reason
            self.log("turn_stopped", reason=reason)

    @property
    def stopped(self) -> bool:
        return self.stop_reason is not None

    @property
    def writes(self) -> list[dict[str, Any]]:
        """The writes executed in this turn, first result of each (a replay is not a second write)."""
        return [item["result"] for item in self.evidence
                if item["tool"] in WRITE_ACTIONS
                and item["result"].get("status") == "executed" and not item["result"].get("replayed")]


def parse_json(content: str | None, model: type[BaseModel] | None = None) -> Any:
    """Parse a JSON reply, optionally into a model. Raises ValueError on bad output."""
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply is not JSON: {exc.msg}") from exc
    if model is None:
        return data
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"reply does not match {model.__name__}: {exc.errors(include_url=False)}") from exc


def chat_json(ctx: TurnContext, role: str, messages: list[dict[str, Any]], model: type[BaseModel], *,
              before_call: Callable[[], None], prepare: Callable[[Any], Any] = lambda data: data) -> BaseModel:
    """A schema-constrained JSON call parsed into `model`. A malformed reply is sent back with the validation error
    for another try, `retries.invalid_json` times; then the last ValueError is raised. `before_call`
    checks the budget before each try; `prepare` adjusts the parsed JSON before validation."""
    messages, attempts = list(messages), ctx.retries.invalid_json + 1
    for attempt in range(1, attempts + 1):
        before_call()
        reply = ctx.chat(role, messages, response_schema=model.model_json_schema())
        try:
            return parse_json(json.dumps(prepare(parse_json(reply.content))), model)
        except (ValueError, TypeError) as exc:
            ctx.log("json_retry" if attempt < attempts else "json_failed", role=role, attempt=attempt, error=str(exc))
            if attempt == attempts:
                raise ValueError(str(exc)) from exc
            messages += [{"role": "assistant", "content": reply.content or ""},
                         {"role": "user", "content": f"That reply was invalid: {exc}. Return only the JSON "
                                                     "object, with exactly the fields of the schema."}]
    raise AssertionError("unreachable")
