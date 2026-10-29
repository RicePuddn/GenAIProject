"""HTTP API for the multi-agent arm, alongside the single-agent `api.py` (which is unchanged).

    uvicorn support_ops.api_multi:app --reload --host 127.0.0.1 --port 8001

This is S3 (multi-agent with RAG policy). The config comes from SUPPORT_OPS_MULTI_CONFIG
(default configs/experiments/multi_agent_rag.yaml). Same endpoints and streaming as `api.py`.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from support_ops.helpers.clock import Clock
from support_ops.helpers.config import load_config
from support_ops.agents.runtime import llm_from_config
from support_ops.helpers.llm import LLM
from support_ops.helpers.tracing import Tracer
from support_ops.policy.rules import Rules
from support_ops.systems.multi_agent import MultiAgentWorkflow
from support_ops.policy.rag import RagPolicy

log = logging.getLogger("uvicorn.error")
DEFAULT_CONFIG = "configs/experiments/multi_agent_rag.yaml"


class Turn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    role: Literal["customer", "agent"]
    content: str = Field(min_length=1, max_length=20_000)


class MultiChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    customer_message: str = Field(min_length=1, max_length=20_000)
    ticket_id: str | None = Field(default=None, min_length=1, max_length=100)
    strategy: Literal["react", "plan_and_execute", "reflexion"] | None = None
    history: list[Turn] = Field(default_factory=list, max_length=50)


class MultiChatResponse(BaseModel):
    ticket_id: str
    strategy: str
    reply: str
    ticket_status: str
    triage: dict[str, Any] | None = None
    proposal: dict[str, Any] | None = None
    min_route: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None
    reply_review: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    handoff_packet: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    reflections: list[str] = Field(default_factory=list)
    audit_log: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_count: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    trace: dict[str, Any]
    history: list[Turn] = Field(default_factory=list)   # send back as-is with the next message


def config_path() -> str:
    return os.environ.get("SUPPORT_OPS_MULTI_CONFIG", DEFAULT_CONFIG)


@lru_cache(maxsize=1)
def _runtime() -> tuple[dict[str, Any], LLM]:
    cfg = load_config(config_path())
    return cfg, llm_from_config(cfg)


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg = load_config(config_path())
    started = time.perf_counter()
    RagPolicy.from_config(cfg, Rules.load(cfg["policy"]["rules_file"])).warm_up()
    log.info("multi-agent %s, policy mode %r ready in %.1fs", config_path(), cfg["policy"]["mode"],
             time.perf_counter() - started)
    yield


app = FastAPI(title="Policy-Bound Support Ops (multi-agent)", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, Any]:
    cfg = load_config(config_path())
    policy = cfg["policy"]
    return {
        "system": "multi_agent", "config": config_path(), "policy_mode": policy["mode"],
        "corpus": policy["corpus_size"],
        "default_strategy": cfg.get("agent", {}).get("strategy", "react"),
        "models": {role: cfg["llm"]["models"][role] for role in ("supervisor", "specialist", "approver")},
        "providers": {role: {**cfg["llm"], **cfg.get("multi_agent", {}).get("llm_by_role", {}).get(role, {})}["provider"]
                      for role in ("guard", "supervisor", "specialist", "approver")},
        "reasoning_effort": cfg["llm"].get("reasoning_effort"),
    }


def _run_chat(request: MultiChatRequest, tracer: Tracer) -> MultiChatResponse:
    cfg, llm = _runtime()
    # A fresh workflow per request keeps ticket state, tokens and the tracer isolated.
    workflow = MultiAgentWorkflow(llm, cfg, clock=Clock(datetime.now(timezone.utc)))
    history = [turn.model_dump() for turn in request.history]
    result = workflow.invoke(tracer.ticket_id, request.customer_message, strategy=request.strategy,
                             tracer=tracer, history=history)
    tracer = result.pop("trace")
    reply = result.pop("reply")
    result.pop("history")
    return MultiChatResponse(
        **result, reply=reply, tool_call_count=tracer.tool_call_count, tool_calls=tracer.tool_calls,
        trace=tracer.summary(),
        history=[*history, {"role": "customer", "content": request.customer_message},
                 *([{"role": "agent", "content": reply}] if reply else [])],
    )


def _ticket_id(request: MultiChatRequest) -> str:
    return request.ticket_id or f"TKT-{uuid4().hex[:12].upper()}"


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/chat", response_model=MultiChatResponse)
def chat(request: MultiChatRequest) -> MultiChatResponse:
    try:
        return _run_chat(request, Tracer(_ticket_id(request)))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/chat/stream")
def chat_stream(request: MultiChatRequest) -> StreamingResponse:
    """The same turn as /chat as Server-Sent Events: `trace` events, then `result` or `error`."""
    events: queue.Queue = queue.Queue()
    finished = object()

    def work() -> None:
        try:
            tracer = Tracer(_ticket_id(request), listener=lambda event: events.put(("trace", event)))
            events.put(("result", _run_chat(request, tracer).model_dump(mode="json")))
        except Exception as exc:
            events.put(("error", {"detail": str(exc)}))
        finally:
            events.put(finished)

    def stream():
        threading.Thread(target=work, daemon=True).start()
        while (item := events.get()) is not finished:
            yield _sse(*item)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
