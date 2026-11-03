"""LLM access behind one small interface for Ollama and Azure OpenAI."""

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
    """The configured provider failed after the client's retries."""


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
        llm = cfg["llm"]
        self.provider = llm["provider"]
        if self.provider not in {"ollama", "azure_openai"}:
            raise ValueError(f"unsupported LLM provider {self.provider!r}")
        self.models: dict[str, str] = llm["models"]
        self.pricing: dict[str, dict[str, float]] = llm["pricing"]
        self.temperature = llm["temperature"]
        self.temperature_by_model: dict[str, float | None] = llm.get("temperature_by_model", {})
        self.max_tokens = llm["max_tokens"]
        self.max_tokens_by_role: dict[str, int] = llm["max_tokens_by_role"]
        self.cache_enabled_by_role: dict[str, bool] = llm["cache_enabled_by_role"]
        self.azure_deployment_envs: dict[str, str] = llm.get("deployment_envs", {})
        self.azure_deployment_defaults: dict[str, str] = llm.get("deployment_defaults", {})
        self.completion_token_models = set(llm.get("completion_token_models", []))
        self.reasoning_effort_by_role: dict[str, str] = llm.get("reasoning_effort_by_role", {})
        self.reasoning_effort_by_model: dict[str, str] = llm.get("reasoning_effort_by_model", {})
        self.no_parallel_tools_models = set(llm.get("no_parallel_tools_models", []))
        self.seed = cfg["run"]["seed"]
        # With temperature 0 and a fixed seed, the request alone would make every trial replay the first.
        self.trial = cfg["run"]["trial"]
        cache_dir = llm["cache_dir"]
        self.cache = (ROOT / cache_dir).resolve() if cache_dir else None
        api_key_env = llm.get("api_key_env")
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if not api_key and self.provider == "ollama":
            # Ollama's local OpenAI-compatible endpoint requires a value the server ignores.
            api_key = "ollama"
        if self.provider == "ollama":
            from openai import OpenAI

            self.client = OpenAI(base_url=llm["base_url"], api_key=api_key, max_retries=4)
        else:
            from openai import AzureOpenAI

            if not api_key:
                raise RuntimeError(f"set {api_key_env or 'the configured Azure API key'} (see .env.example)")
            endpoint_env, version_env = llm["azure_endpoint_env"], llm["api_version_env"]
            endpoint, api_version = os.environ.get(endpoint_env), os.environ.get(version_env)
            missing = [name for name, value in ((endpoint_env, endpoint), (version_env, api_version)) if not value]
            if missing:
                raise RuntimeError("set the configured Azure variables: " + ", ".join(missing))
            apim_header_env = llm.get("apim_header_env")
            apim_key = os.environ.get(apim_header_env) if apim_header_env else None
            headers = {"Ocp-Apim-Subscription-Key": apim_key} if apim_key else None
            self.client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version,
                                      default_headers=headers, max_retries=4)

    def _cost(self, model: str, tin: int, tout: int) -> float:
        p = self.pricing.get(model)
        return 0.0 if p is None else (tin * p["input"] + tout * p["output"]) / 1e6

    def chat(self, role, messages, tools=None, tracer=None, json_mode=False) -> ChatResult:
        model = self.models[role]
        provider_model = model
        if self.provider == "azure_openai":
            deployment_env = self.azure_deployment_envs.get(model)
            provider_model = os.environ.get(deployment_env) if deployment_env else None
            provider_model = provider_model or self.azure_deployment_defaults.get(model)
            if not provider_model:
                raise RuntimeError(f"Azure deployment for model {model!r} is not configured")
        req: dict[str, Any] = {"model": provider_model, "messages": messages}
        temperature = self.temperature_by_model.get(model, self.temperature)
        if temperature is not None:
            req["temperature"] = temperature
        max_tokens = self.max_tokens_by_role.get(role, self.max_tokens)
        if self.provider == "azure_openai" and model in self.completion_token_models:
            req["max_completion_tokens"] = max_tokens
            effort = self.reasoning_effort_by_role.get(role) or self.reasoning_effort_by_model.get(model)
            if effort:
                req["reasoning_effort"] = effort
                if tools and model in self.no_parallel_tools_models:
                    req["parallel_tool_calls"] = False
        else:
            req["max_tokens"] = max_tokens
        if self.provider == "ollama":
            req["seed"] = self.seed
        if tools:
            req["tools"] = tools
        if json_mode:
            req["response_format"] = {"type": "json_object"}
        key = hashlib.sha256(json.dumps({**req, "trial": self.trial}, sort_keys=True, default=str).encode()).hexdigest()
        cache_enabled = self.cache_enabled_by_role.get(role, True)
        cached = self._cache_get(key) if cache_enabled else None
        t0 = time.perf_counter()
        if cached is None:
            try:
                resp = self.client.chat.completions.create(**req)
            except Exception as exc:
                from openai import OpenAIError

                if isinstance(exc, OpenAIError):
                    raise ProviderError(type(exc).__name__) from exc
                raise
            else:
                msg = resp.choices[0].message
                calls = [
                    ToolCall(tc.id, tc.function.name, _parse_args(tc.function.arguments))
                    for tc in (msg.tool_calls or [])
                ]
                usage = resp.usage
                tin = usage.prompt_tokens if usage else count_tokens(json.dumps(req["messages"]))
                tout = usage.completion_tokens if usage else count_tokens(msg.content or "")
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
