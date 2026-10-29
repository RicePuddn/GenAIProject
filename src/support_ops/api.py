"""HTTP API for interactive support-flow trials."""
from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from support_ops.helpers.clock import Clock
from support_ops.helpers.config import load_config
from support_ops.helpers.llm import OpenAICompatLLM
from support_ops.helpers.tracing import Tracer
from support_ops.systems.single_agent import SingleAgentWorkflow

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
    history: list[Turn] = Field(default_factory=list, max_length=50)


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
    history: list[Turn] = Field(default_factory=list)   # send back as-is with the next message


@lru_cache(maxsize=1)
def _runtime() -> tuple[dict[str, Any], OpenAICompatLLM]:
    cfg = load_config()
    return cfg, OpenAICompatLLM(cfg)


app = FastAPI(title="Policy-Bound Support Ops", version="0.1.0")

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


def _run_chat(request: ChatRequest, tracer: Tracer) -> ChatResponse:
    """One turn of the workflow; shared by /chat and /chat/stream."""
    cfg, llm = _runtime()
    # Keep mutable workflow state and its per-ticket tracer isolated per request.
    workflow = SingleAgentWorkflow(llm, cfg, clock=Clock(datetime.now(timezone.utc)))
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
        history=[*history, {"role": "customer", "content": request.customer_message},
                 *([{"role": "agent", "content": reply}] if reply else [])],
    )


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


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """The same turn as /chat, answered as Server-Sent Events while it runs: one `trace` event
    per step (guard, thought, tool call, tool result, handoff), then a `result` event holding
    the /chat response, or an `error` event. One stateless POST, as in MCP's Streamable HTTP."""
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
