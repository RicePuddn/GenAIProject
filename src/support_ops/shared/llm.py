"""LLM access behind one small interface.

`OpenAICompatLLM` talks to any OpenAI-compatible endpoint (Groq by default), so the
model family can be swapped from config without touching agent code. Tests use the scripted
double in tests/fakes.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

import tiktoken

from .config import ROOT
from .tracing import Tracer

_ENC = tiktoken.get_encoding("o200k_base")


class ProviderError(Exception):
    """The LLM provider failed after the client's own retries (rate limit, outage, rejected request). No
    agent can fix it, so it ends the turn, and the ticket waits to be sent again."""


def count_tokens(text: str) -> int:
    # Customer text may contain special-token strings such as <|endoftext|>; count them as plain text.
    return len(_ENC.encode(text, disallowed_special=()))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning: str | None = None   

    def as_message(self) -> dict[str, Any]:
        """The assistant message to append to the conversation."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in self.tool_calls
            ]
        return msg


class LLM(Protocol):
    def chat(self, role: str, messages: list[dict[str, Any]], tools: list[dict] | None = None,
             tracer: Tracer | None = None, json_mode: bool = False) -> ChatResult: ...


class OpenAICompatLLM:
    def __init__(self, cfg: dict):
        from openai import OpenAI

        llm = cfg["llm"]
        self.models: dict[str, str] = llm["models"]
        self.pricing: dict[str, dict[str, float]] = llm["pricing"]
        self.temperature = llm["temperature"]
        self.reasoning_effort = llm["reasoning_effort"]
        self.max_tokens = llm["max_tokens"]
        self.max_tokens_by_role: dict[str, int] = llm["max_tokens_by_role"]
        self.cache_enabled_by_role: dict[str, bool] = llm["cache_enabled_by_role"]
        self.seed = cfg["run"]["seed"]
        # With temperature 0 and a fixed seed, the request alone would make every trial replay the first.
        self.trial = cfg["run"]["trial"]
        cache_dir = llm["cache_dir"]
        self.cache = (ROOT / cache_dir).resolve() if cache_dir else None
        api_key = os.environ.get(llm["api_key_env"])
        if not api_key:
            raise RuntimeError(f"set {llm['api_key_env']} (see .env.example)")
        self.client = OpenAI(base_url=llm["base_url"], api_key=api_key, max_retries=4)

    def _cost(self, model: str, tin: int, tout: int) -> float:
        p = self.pricing.get(model)
        return 0.0 if p is None else (tin * p["input"] + tout * p["output"]) / 1e6

    def chat(self, role, messages, tools=None, tracer=None, json_mode=False) -> ChatResult:
        model = self.models[role]
        req: dict[str, Any] = {
            "model": model, "messages": messages, "temperature": self.temperature,
            "max_tokens": self.max_tokens_by_role.get(role, self.max_tokens), "seed": self.seed,
        }
        if tools:
            req["tools"] = tools
        if json_mode:
            req["response_format"] = {"type": "json_object"}
        if self.reasoning_effort and "gpt-oss" in model:
            req["reasoning_effort"] = self.reasoning_effort

        key = hashlib.sha256(json.dumps({**req, "trial": self.trial}, sort_keys=True, default=str).encode()).hexdigest()
        cache_enabled = self.cache_enabled_by_role.get(role, True)
        cached = self._cache_get(key) if cache_enabled else None
        t0 = time.perf_counter()
        if cached is None:
            try:
                resp = self.client.chat.completions.create(**req)
            except Exception as exc:
                recovered = _recover_tool_calls(exc, key) if tools else None
                rejected_json = _rejected_json(exc) if json_mode and recovered is None else None
                if recovered is None and rejected_json is None:
                    from openai import OpenAIError

                    if isinstance(exc, OpenAIError):
                        raise ProviderError(type(exc).__name__) from exc
                    raise
                # Not cached: the provider rejected it, so the same request may succeed next time. The
                # estimate counts what was sent (messages and tool schemas) and what it generated.
                sent = count_tokens(json.dumps([messages, tools], default=str))
                if recovered is not None:
                    result = ChatResult(None, recovered, sent, count_tokens(json.dumps(
                        [[call.name, call.arguments] for call in recovered], default=str)))
                    if tracer is not None:
                        tracer.log("tool_call_recovered", role=role, tools=[call.name for call in recovered])
                else:
                    result = ChatResult(rejected_json, [], sent, count_tokens(rejected_json))
                    if tracer is not None:
                        tracer.log("json_rejected_by_provider", role=role)
            else:
                msg = resp.choices[0].message
                calls = [
                    ToolCall(tc.id, tc.function.name, _parse_args(tc.function.arguments))
                    for tc in (msg.tool_calls or [])
                ]
                usage = resp.usage
                tin = usage.prompt_tokens if usage else count_tokens(json.dumps(messages))
                tout = usage.completion_tokens if usage else count_tokens(msg.content or "")
                # gpt-oss on Groq returns its thinking in a provider-specific `reasoning` field.
                result = ChatResult(msg.content, calls, tin, tout, getattr(msg, "reasoning", None) or None)
                if cache_enabled:
                    self._cache_put(key, result)
        else:
            result = cached
        if tracer is not None:
            tracer.record_llm(role, model, result.input_tokens, result.output_tokens,
                              self._cost(model, result.input_tokens, result.output_tokens) if cached is None else 0.0,
                              time.perf_counter() - t0, cached is not None)
        return result

    def _cache_get(self, key: str) -> ChatResult | None:
        if self.cache is None:
            return None
        path = self.cache / f"{key}.json"
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            result = ChatResult(d["content"], [ToolCall(**tc) for tc in d["tool_calls"]],
                                d["input_tokens"], d["output_tokens"], d.get("reasoning"))
        except (OSError, ValueError, KeyError, TypeError):
            return None
        # Old/invalid empty responses must never become a permanent cache hit.
        return result if result.content or result.tool_calls else None

    def _cache_put(self, key: str, r: ChatResult) -> None:
        if self.cache is None or (not r.content and not r.tool_calls):
            return
        self.cache.mkdir(parents=True, exist_ok=True)
        payload = {
            "content": r.content, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "tool_calls": [tc.__dict__ for tc in r.tool_calls], "reasoning": r.reasoning,
        }
        # Readers only see a complete JSON file, including with concurrent API requests.
        with NamedTemporaryFile(mode="w", dir=self.cache, delete=False) as temporary:
            try:
                json.dump(payload, temporary)
                temporary.close()
                os.replace(temporary.name, self.cache / f"{key}.json")
            finally:
                Path(temporary.name).unlink(missing_ok=True)


def _parse_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"_raw": raw}
    except json.JSONDecodeError:
        return {"_raw": raw}  # surfaced to the agent as a tool error, not a crash


def _error_body(exc: Exception) -> dict[str, Any] | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    return body if isinstance(body, dict) else None


def _rejected_json(exc: Exception) -> str | None:
    """Groq checks a json_mode reply and rejects the whole response (400 json_validate_failed) when it is
    not valid JSON. The error carries what the model wrote, so it comes back as the reply: the caller's
    own parsing then reports it as invalid and asks again, as for any malformed reply. None for any
    other error."""
    body = _error_body(exc)
    if body is None or body.get("code") != "json_validate_failed":
        return None
    return str(body.get("failed_generation") or "")


def _recover_tool_calls(exc: Exception, request_key: str = "") -> list[ToolCall] | None:
    """Groq checks tool arguments against the schema and rejects the whole response (400 tool_use_failed)
    on a mismatch, e.g. an alias such as order_item_id. The error carries the attempted call, so it comes
    back as an ordinary tool call: the tool's own validation then accepts it or returns an error the agent
    can fix. None for any other error. IDs come from `request_key` (the request's hash) and the call's
    position, so a replayed ticket gets the same IDs and its later turns still hit the cache."""
    body = _error_body(exc)
    if body is None or body.get("code") != "tool_use_failed":
        return None
    try:
        generated = json.loads(body.get("failed_generation") or "")
    except (TypeError, json.JSONDecodeError):
        return None
    calls = []
    for index, attempt in enumerate(generated if isinstance(generated, list) else [generated]):
        if not isinstance(attempt, dict) or not isinstance(attempt.get("name"), str):
            return None
        # Some generations name the arguments "parameters".
        arguments = attempt.get("arguments", attempt.get("parameters", {}))
        if isinstance(arguments, str):
            arguments = _parse_args(arguments)
        elif not isinstance(arguments, dict):   # e.g. a list or number: the tool reports it as an error
            arguments = {"_raw": json.dumps(arguments, default=str)}
        call_id = hashlib.sha256(f"{request_key}:{index}".encode()).hexdigest()[:12]
        calls.append(ToolCall(f"recovered_{call_id}", attempt["name"], arguments))
    return calls or None
