"""What every multi-agent role shares within one ticket: the LLM call, the trace and the budget."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from support_ops.helpers.clock import Clock
from support_ops.helpers.config import deep_merge
from support_ops.helpers.llm import LLM, ChatResult, OpenAICompatLLM
from support_ops.helpers.tracing import Tracer


class RoleRoutedLLM:
    """Sends each role to its own OpenAI-compatible client. Roles without an override use the
    default client, so a config without `multi_agent.llm_by_role` has a single client."""

    def __init__(self, default: LLM, by_role: dict[str, LLM]):
        self.default, self.by_role = default, by_role

    def chat(self, role: str, messages: list[dict[str, Any]], tools: list[dict] | None = None,
             tracer: Tracer | None = None, json_mode: bool = False) -> ChatResult:
        return self.by_role.get(role, self.default).chat(role, messages, tools=tools, tracer=tracer,
                                                         json_mode=json_mode)


def llm_from_config(cfg: dict[str, Any]) -> LLM:
    """`llm` sets the default provider; `multi_agent.llm_by_role.<role>` overrides `llm` keys per role."""
    overrides = cfg.get("multi_agent", {}).get("llm_by_role", {})
    default = OpenAICompatLLM(cfg)
    if not overrides:
        return default
    return RoleRoutedLLM(default, {role: OpenAICompatLLM(deep_merge(cfg, {"llm": override}))
                                   for role, override in overrides.items()})


class BudgetExceeded(Exception):
    """A per-ticket limit was reached; the ticket goes to a human."""


@dataclass
class Budget:
    """Ticket-wide limits from `limits` in config, checked before each LLM or tool call."""
    limits: dict[str, Any]
    tracer: Tracer
    tool_steps: int = 0

    def check(self) -> None:
        if self.tool_steps >= self.limits["max_steps_per_ticket"]:
            raise BudgetExceeded("tool execution limit reached")
        if self.tracer.cost_usd >= self.limits["max_cost_usd_per_ticket"]:
            raise BudgetExceeded("LLM cost budget reached")


@dataclass(frozen=True)
class Retries:
    """`multi_agent.retries` in config."""
    tool_errors: int = 1     # automatic re-runs of a tool that failed for a reason the model can't fix
    invalid_json: int = 1    # extra tries for a malformed JSON reply (supervisor, approver)
    backoff_s: float = 0.5   # wait before re-running a tool, times the attempt number

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Retries:
        values = cfg.get("multi_agent", {}).get("retries", {})
        if unknown := set(values) - {"tool_errors", "invalid_json", "backoff_s"}:
            raise ValueError(f"unknown multi_agent.retries settings: {sorted(unknown)}")
        retries = cls(**values)
        if retries.tool_errors < 0 or retries.invalid_json < 0 or retries.backoff_s < 0:
            raise ValueError("multi_agent.retries values must not be negative")
        return retries


@dataclass
class AgentContext:
    chat: Callable[..., ChatResult]   # (role, messages, tools=None, json_mode=False)
    log: Callable[..., None]          # (kind, **payload) into the ticket's trace
    clock: Clock
    budget: Budget
    retries: Retries = field(default_factory=Retries)
    _calls: list[int] = field(default_factory=lambda: [0])

    def next_call_id(self) -> str:
        self._calls[0] += 1
        return f"call-{self._calls[0]}"


def chat_json(ctx: AgentContext, role: str, messages: list[dict[str, Any]], model: type[BaseModel],
              prepare: Callable[[Any], Any] = lambda data: data) -> BaseModel:
    """A json_mode call parsed into `model`. A malformed reply is sent back with the validation error
    for another try, `retries.invalid_json` times; then the last ValueError is raised. `prepare`
    adjusts the parsed JSON before validation (e.g. to add fields set by code)."""
    messages, attempts = list(messages), ctx.retries.invalid_json + 1
    for attempt in range(1, attempts + 1):
        ctx.budget.check()
        reply = ctx.chat(role, messages, json_mode=True)
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


def parse_json(content: str | None, model: type[BaseModel] | None = None) -> Any:
    """Parse a json_mode reply, optionally into a model. Raises ValueError on bad output."""
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
