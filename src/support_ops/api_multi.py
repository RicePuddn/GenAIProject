"""HTTP API for the multi-agent arm, alongside the single-agent `api.py`.

    uvicorn support_ops.api_multi:app --reload --host 127.0.0.1 --port 8001

The config comes from SUPPORT_OPS_MULTI_CONFIG: configs/experiments/multi_agent_rag.yaml (S3, the
default) or configs/experiments/multi_agent_skills.yaml (S4). Same endpoints and streaming as `api.py`:
streams run on a bounded worker pool, and a client that disconnects cancels its ticket at the
next workflow step.
"""
from __future__ import annotations

import asyncio
import logging
import os
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

from support_ops.agents.runtime import llm_from_config
from support_ops.api import MAX_HISTORY_TURNS, CancellableStreamingResponse, Turn, _retain_history, _sse
from support_ops.helpers.cancellation import CancellationToken, WorkflowCancelled
from support_ops.helpers.clock import Clock
from support_ops.helpers.config import load_config
from support_ops.helpers.llm import LLM
from support_ops.helpers.tracing import Tracer
from support_ops.helpers.workers import WorkflowWorkers
from support_ops.policy.rules import Rules
from support_ops.systems.multi_agent import MultiAgentWorkflow
from support_ops.systems.single_agent import policy_from_config
from support_ops.systems.supervisor_agent import SupervisorAgentWorkflow

log = logging.getLogger("uvicorn.error")
DEFAULT_CONFIG = "configs/experiments/multi_agent_rag.yaml"


class MultiChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    customer_message: str = Field(min_length=1, max_length=20_000)
    ticket_id: str | None = Field(default=None, min_length=1, max_length=100)
    strategy: Literal["react", "plan_and_execute"] | None = None        # the specialists
    orchestration: Literal["react", "plan_and_execute"] | None = None   # the supervisor; default: config, else strategy
    # workflow: fixed steps (MultiAgentWorkflow); supervisor: the supervisor chooses every step. Default: config.
    design: Literal["workflow", "supervisor"] | None = None
    # Stateless: the client sends the earlier turns of the ticket; the server keeps nothing.
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


class MultiChatResponse(BaseModel):
    ticket_id: str
    strategy: str
    orchestration: str
    design: str = "workflow"
    reply: str
    ticket_status: str
    triage: dict[str, Any] | None = None
    proposal: dict[str, Any] | None = None
    min_route: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None
    reply_review: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    handoff_packet: dict[str, Any] | None = None
    # One per specialist when triage split the message; proposal etc. above are the last part's.
    parts: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    audit_log: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_count: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    trace: dict[str, Any]
    # Send back as-is with the next message; trimmed to the newest exchanges that fit the limit.
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


def config_path() -> str:
    return os.environ.get("SUPPORT_OPS_MULTI_CONFIG", DEFAULT_CONFIG)


@lru_cache(maxsize=1)
def _runtime() -> tuple[dict[str, Any], LLM]:
    cfg = load_config(config_path())
    return cfg, llm_from_config(cfg)


@lru_cache(maxsize=1)
def _stream_workers() -> WorkflowWorkers:
    """This app's own pool, so S3 streams never take slots from the single-agent API."""
    return WorkflowWorkers()


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg = load_config(config_path())
    started = time.perf_counter()
    policy_from_config(cfg, Rules.load(cfg["policy"]["rules_file"])).warm_up()
    log.info("multi-agent %s, policy mode %r ready in %.1fs", config_path(), cfg["policy"]["mode"],
             time.perf_counter() - started)
    workers = _stream_workers()
    try:
        yield
    finally:
        workers.close()
        _stream_workers.cache_clear()


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
        "corpus": policy["skills_dir"] if policy["mode"] == "skills" else policy["corpus_size"],
        "default_strategy": cfg.get("agent", {}).get("strategy", "react"),
        "default_orchestration": cfg.get("multi_agent", {}).get("orchestration")
        or cfg.get("agent", {}).get("strategy", "react"),
        "default_design": cfg.get("multi_agent", {}).get("design", "workflow"),
        "models": {role: cfg["llm"]["models"][role] for role in ("supervisor", "specialist", "approver")},
        "providers": {role: {**cfg["llm"], **cfg.get("multi_agent", {}).get("llm_by_role", {}).get(role, {})}["provider"]
                      for role in ("guard", "supervisor", "specialist", "approver")},
        "reasoning_effort": cfg["llm"].get("reasoning_effort"),
    }


def _run_chat(request: MultiChatRequest, tracer: Tracer,
              cancellation: CancellationToken | None = None) -> MultiChatResponse:
    """One turn of the workflow; shared by /chat and /chat/stream."""
    cancellation = cancellation or CancellationToken()
    cancellation.check()
    cfg, llm = _runtime()
    # A fresh workflow per request keeps ticket state, tokens and the tracer isolated.
    design = request.design or cfg.get("multi_agent", {}).get("design", "workflow")
    system = SupervisorAgentWorkflow if design == "supervisor" else MultiAgentWorkflow
    workflow = system(llm, cfg, clock=Clock(datetime.now(timezone.utc)), cancellation=cancellation)
    history = [turn.model_dump() for turn in request.history]
    result = workflow.invoke(tracer.ticket_id, request.customer_message, strategy=request.strategy,
                             tracer=tracer, history=history, orchestration=request.orchestration)
    tracer = result.pop("trace")
    reply = result.pop("reply")
    result.pop("history")
    return MultiChatResponse(
        **result, reply=reply, tool_call_count=tracer.tool_call_count, tool_calls=tracer.tool_calls,
        trace=tracer.summary(),
        history=_retain_history([*history, {"role": "customer", "content": request.customer_message},
                                 *([{"role": "agent", "content": reply}] if reply else [])]),
    )


def _ticket_id(request: MultiChatRequest) -> str:
    return request.ticket_id or f"TKT-{uuid4().hex[:12].upper()}"


@app.post("/chat", response_model=MultiChatResponse)
def chat(request: MultiChatRequest) -> MultiChatResponse:
    try:
        return _run_chat(request, Tracer(_ticket_id(request)))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/chat/stream")
def chat_stream(request: MultiChatRequest) -> StreamingResponse:
    """The same turn as /chat as Server-Sent Events (`trace` events, then `result` or `error`),
    on a bounded, cancellable worker, as in `api.py`."""
    events: queue.Queue = queue.Queue(maxsize=128)
    finished = object()
    cancellation = CancellationToken()

    def publish(item) -> None:
        # A full queue means the client is slow; keep waiting unless it has gone away.
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
