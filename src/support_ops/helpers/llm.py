"""LLM access behind one small interface.

`OpenAICompatLLM` talks to any OpenAI-compatible endpoint (Groq by default), so the
model family can be swapped from config without touching agent code. `ScriptedLLM`
returns pre-written responses, which lets the whole pipeline run in tests with no API key.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Protocol

import tiktoken

from .config import ROOT
from .tracing import Tracer

_ENC = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


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
        self.pricing: dict[str, dict[str, float]] = llm.get("pricing", {})
        self.temperature = llm.get("temperature", 0.0)
        self.reasoning_effort = llm.get("reasoning_effort")
        self.max_tokens = llm.get("max_tokens", 2048)
        self.max_tokens_by_role: dict[str, int] = llm.get("max_tokens_by_role", {})
        self.cache_enabled_by_role: dict[str, bool] = llm.get("cache_enabled_by_role", {})
        self.seed = cfg["run"]["seed"]
        cache_dir = llm.get("cache_dir")
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

        key = hashlib.sha256(json.dumps(req, sort_keys=True, default=str).encode()).hexdigest()
        cache_enabled = self.cache_enabled_by_role.get(role, True)
        cached = self._cache_get(key) if cache_enabled else None
        t0 = time.perf_counter()
        if cached is None:
            resp = self.client.chat.completions.create(**req)
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


Responder = Callable[[list[dict[str, Any]], list[dict] | None], ChatResult]


class ScriptedLLM:
    """Test double. Register, per role, either a queue of ChatResults or a function
    that builds one from the conversation. Token counts are estimated with tiktoken."""

    def __init__(self) -> None:
        self.queues: dict[str, deque[ChatResult]] = defaultdict(deque)
        self.responders: dict[str, Responder] = {}
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def push(self, role: str, *results: ChatResult) -> "ScriptedLLM":
        self.queues[role].extend(results)
        return self

    def on(self, role: str, fn: Responder) -> "ScriptedLLM":
        self.responders[role] = fn
        return self

    def chat(self, role, messages, tools=None, tracer=None, json_mode=False) -> ChatResult:
        self.calls.append((role, messages))
        if self.queues[role]:
            r = self.queues[role].popleft()
        elif role in self.responders:
            r = self.responders[role](messages, tools)
        else:
            raise AssertionError(f"ScriptedLLM has no response left for role {role!r}")
        r.input_tokens = r.input_tokens or count_tokens(json.dumps(messages, default=str))
        r.output_tokens = r.output_tokens or count_tokens(
            (r.content or "") + json.dumps([tc.arguments for tc in r.tool_calls]))
        if tracer is not None:
            tracer.record_llm(role, "scripted", r.input_tokens, r.output_tokens, 0.0, 0.0, False)
        return r


def tool_call(name: str, _id: str | None = None, **arguments: Any) -> ChatResult:
    """Shorthand for scripting a single tool call in tests."""
    return ChatResult(None, [ToolCall(_id or f"call_{name}", name, arguments)])


def text(content: str) -> ChatResult:
    return ChatResult(content)
