"""HTTP API for interactive support-flow trials."""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from support_ops.helpers.cancellation import CancellationToken, WorkflowCancelled
from support_ops.helpers.clock import Clock
from support_ops.helpers.config import load_config
from support_ops.helpers.llm import OpenAICompatLLM
from support_ops.helpers.tracing import Tracer
from support_ops.helpers.workers import WorkflowWorkers
from support_ops.policy.rules import Rules
from support_ops.systems.single_agent import SingleAgentWorkflow, policy_from_config

MAX_HISTORY_TURNS = 50

log = logging.getLogger("uvicorn.error")   # shown in the uvicorn console


class Turn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    role: Literal["customer", "agent"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    customer_message: str = Field(min_length=1, max_length=20_000)
    ticket_id: str | None = Field(default=None, min_length=1, max_length=100)
    strategy: Literal["react", "plan_and_execute"] | None = None
    # Stateless: the client sends the earlier turns of the ticket; the server keeps nothing.
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


class ChatResponse(BaseModel):
    ticket_id: str
    strategy: str
    reply: str
    ticket_status: str
    handoff_packet: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_count: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    trace: dict[str, Any]
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


@lru_cache(maxsize=1)
def _runtime() -> tuple[dict[str, Any], OpenAICompatLLM]:
    cfg = load_config()
    return cfg, OpenAICompatLLM(cfg)


@lru_cache(maxsize=1)
def _stream_workers() -> WorkflowWorkers:
    return WorkflowWorkers()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pay one-off policy setup (RAG models, corpus embeddings) at startup, not in the first ticket.
    cfg = load_config()
    started = time.perf_counter()
    policy_from_config(cfg, Rules.load(cfg["policy"]["rules_file"])).warm_up()
    log.info("policy mode %r ready in %.1fs", cfg["policy"].get("mode", "skills"), time.perf_counter() - started)
    workers = _stream_workers()
    try:
        yield
    finally:
        workers.close()
        _stream_workers.cache_clear()


app = FastAPI(title="Policy-Bound Support Ops", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, Any]:
    """What this server runs, for clients such as the demo page. Needs no API key or database."""
    cfg = load_config()
    policy = cfg["policy"]
    return {
        "policy_mode": policy.get("mode", "skills"),
        "corpus": policy["corpus_size"] if policy.get("mode") == "rag" else policy["skills_dir"],
        "default_strategy": cfg.get("agent", {}).get("strategy", "react"),
        "model": cfg["llm"]["models"]["supervisor"],
        "reasoning_effort": cfg["llm"].get("reasoning_effort"),
    }


def _run_chat(request: ChatRequest, tracer: Tracer,
              cancellation: CancellationToken | None = None) -> ChatResponse:
    """One turn of the workflow; shared by /chat and /chat/stream."""
    cancellation = cancellation or CancellationToken()
    cancellation.check()
    cfg, llm = _runtime()
    # Keep mutable workflow state and its per-ticket tracer isolated per request.
    workflow = SingleAgentWorkflow(llm, cfg, clock=Clock(datetime.now(timezone.utc)),
                                   cancellation=cancellation)
    history = [turn.model_dump() for turn in request.history]
    result = workflow.invoke(tracer.ticket_id, request.customer_message, strategy=request.strategy,
                             tracer=tracer, history=history)
    tracer = result.pop("trace")
    reply = result.get("reply", "")
    return ChatResponse(
        ticket_id=tracer.ticket_id,
        strategy=result["strategy"],
        reply=reply,
        ticket_status=result.get("ticket_status", "open"),
        handoff_packet=result.get("handoff_packet"),
        evidence=result.get("evidence", []),
        tool_call_count=tracer.tool_call_count,
        tool_calls=tracer.tool_calls,
        trace=tracer.summary(),
        history=_retain_history([*history, {"role": "customer", "content": request.customer_message},
                                 *([{"role": "agent", "content": reply}] if reply else [])]),
    )


def _retain_history(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the newest complete exchanges, within the next request's limit."""
    pairs = [turns[index:index + 2] for index in range(len(turns) - 1)
             if turns[index]["role"] == "customer" and turns[index + 1]["role"] == "agent"]
    return [turn for pair in pairs[-(MAX_HISTORY_TURNS // 2):] for turn in pair]


def _ticket_id(request: ChatRequest) -> str:
    return request.ticket_id or f"TKT-{uuid4().hex[:12].upper()}"


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return _run_chat(request, Tracer(_ticket_id(request)))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


class CancellableStreamingResponse(StreamingResponse):
    def __init__(self, *args, cancellation: CancellationToken, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancellation = cancellation

    async def __call__(self, scope, receive, send):
        # Also handles ASGI send failures, which can happen while the generator
        # is suspended at yield rather than awaiting the next queue item.
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.cancellation.cancel()
            await self.body_iterator.aclose()


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream trace events and a final result with bounded, cancellable workers."""
    events: queue.Queue = queue.Queue(maxsize=128)
    finished = object()
    cancellation = CancellationToken()

    def publish(item) -> None:
        while True:
            cancellation.check()
            try:
                events.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def work() -> None:
        try:
            try:
                tracer = Tracer(_ticket_id(request), listener=lambda event: publish(("trace", event)))
                result = _run_chat(request, tracer, cancellation)
            except WorkflowCancelled:
                raise
            except Exception as exc:
                publish(("error", {"detail": str(exc)}))
            else:
                publish(("result", result.model_dump(mode="json")))
            publish(finished)
        except WorkflowCancelled:
            pass

    try:
        future = _stream_workers().submit(cancellation, work)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    def poll():
        try:
            return events.get(timeout=0.1)
        except queue.Empty:
            return None

    async def stream():
        try:
            while True:
                item = await asyncio.to_thread(poll)
                if item is finished:
                    break
                if item is not None:
                    yield _sse(*item)
        finally:
            cancellation.cancel()
            future.cancel()

    return CancellableStreamingResponse(
        stream(), cancellation=cancellation, media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
