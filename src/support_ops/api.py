"""HTTP API for interactive support-flow trials."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from support_ops.helpers.clock import Clock
from support_ops.helpers.config import load_config
from support_ops.helpers.llm import OpenAICompatLLM
from support_ops.systems.single_agent import SingleAgentWorkflow


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    customer_message: str = Field(min_length=1, max_length=20_000)
    ticket_id: str | None = Field(default=None, min_length=1, max_length=100)
    strategy: Literal["react", "plan_and_execute"] | None = None


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


@lru_cache(maxsize=1)
def _runtime() -> tuple[dict[str, Any], OpenAICompatLLM]:
    cfg = load_config()
    return cfg, OpenAICompatLLM(cfg)


app = FastAPI(title="Policy-Bound Support Ops", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    ticket_id = request.ticket_id or f"TKT-{uuid4().hex[:12].upper()}"
    try:
        cfg, llm = _runtime()
        # Keep mutable workflow state and its per-ticket tracer isolated per request.
        workflow = SingleAgentWorkflow(llm, cfg, clock=Clock(datetime.now(timezone.utc)))
        result = workflow.invoke(ticket_id, request.customer_message, strategy=request.strategy)
        tracer = result.pop("trace")
        return ChatResponse(
            ticket_id=ticket_id,
            strategy=result["strategy"],
            reply=result.get("reply", ""),
            ticket_status=result.get("ticket_status", "open"),
            handoff_packet=result.get("handoff_packet"),
            evidence=result.get("evidence", []),
            tool_call_count=tracer.tool_call_count,
            tool_calls=tracer.tool_calls,
            trace=tracer.summary(),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
