"""One HTTP API for every setup.

    uvicorn support_ops.api:app --reload --host 127.0.0.1 --port 8000

A request names its setup and variant (`setup`, e.g. "s3-c"); without one, `run.setup` in
configs/base.yaml decides. The workflow date is `run.clock_start`, as in experiment runs, unless the
request gives `clock_start` (the demo's "use today's date"). The API is stateless: the client sends the
ticket's earlier turns as `history`. `/chat/stream` runs on a bounded worker pool, and a client that
disconnects cancels its turn at the next step.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from threading import BoundedSemaphore, Lock
from typing import Any, Callable, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from support_ops.setups import build_setup
from support_ops.shared.cancellation import CancellationToken, WorkflowCancelled
from support_ops.shared.clock import Clock
from support_ops.shared.config import load_config, setup_ids
from support_ops.shared.db import PostgresReader
from support_ops.shared.llm import LLM, OpenAICompatLLM
from support_ops.shared.pipeline import run_turn
from support_ops.shared.policy.provider import policy_from_config
from support_ops.shared.rules import Rules
from support_ops.shared.tracing import Tracer

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
    setup: str | None = Field(default=None, pattern=r"^[sS][1-8]-[a-dA-D]$", description="e.g. s3-c; default run.setup")
    # The signed-in account: reads and writes are scoped to it, and the ticket is saved as theirs.
    # Without one, nothing is scoped and the ticket is saved under the placeholder customer.
    customer_id: str | None = Field(default=None, min_length=1, max_length=100)
    # Stateless: the client sends the earlier turns of the ticket; the server keeps nothing.
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    clock_start: AwareDatetime | None = Field(default=None, description="Overrides run.clock_start for this turn.")


class ChatResponse(BaseModel):
    ticket_id: str
    setup: str
    strategy: dict[str, str]
    customer_id: str | None = None
    reply: str
    ticket_status: str
    note: str | None = None
    handoff_packet: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    writes: list[dict[str, Any]] = Field(default_factory=list)
    parts: list[dict[str, Any]] = Field(default_factory=list)   # multi-agent: each specialist's report
    reply_review: dict[str, Any] | None = None
    audit_log: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_count: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    trace: dict[str, Any]
    # Send back as-is with the next message; trimmed to the newest exchanges that fit the limit.
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


class Workers:
    """A bounded worker pool whose pending turns are cancelled when the server shuts down."""

    def __init__(self, capacity: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix="support-stream")
        self._slots = BoundedSemaphore(capacity)
        self._lock = Lock()
        self._tokens: set[CancellationToken] = set()
        self._closed = False

    def submit(self, token: CancellationToken, work: Callable[[], None]) -> Future:
        with self._lock:
            if self._closed or not self._slots.acquire(blocking=False):
                raise RuntimeError("stream capacity reached; retry when another request finishes")
            self._tokens.add(token)
            try:
                future = self._executor.submit(work)
            except Exception:
                self._tokens.remove(token)
                self._slots.release()
                raise

        def finished(_: Future) -> None:
            with self._lock:
                self._tokens.remove(token)
                self._slots.release()

        future.add_done_callback(finished)
        return future

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for token in self._tokens:
                token.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)


@lru_cache(maxsize=1)
def _llm() -> LLM:
    return OpenAICompatLLM(load_config())


@lru_cache(maxsize=1)
def _stream_workers() -> Workers:
    return Workers()


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg = load_config()
    started = time.perf_counter()
    policy_from_config(cfg, Rules.load(cfg["policy"]["rules_file"])).warm_up()
    log.info("default setup %s, policy mode %r ready in %.1fs", cfg["setup"]["id"], cfg["policy"]["mode"],
             time.perf_counter() - started)
    workers = _stream_workers()
    try:
        yield
    finally:
        workers.close()
        _stream_workers.cache_clear()


app = FastAPI(title="Policy-Bound Support Ops", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, Any]:
    """What this server runs, for clients such as the demo page. Needs no API key or database."""
    cfg = load_config()
    setups = {}
    for setup_id in setup_ids():
        resolved = load_config(setup=setup_id)
        setups[setup_id] = {**{key: resolved["setup"][key] for key in ("name", "variant", "agents", "control")},
                            "policy_mode": resolved["policy"]["mode"], "strategy": resolved["strategy"]}
    return {"default_setup": cfg["setup"]["id"], "setups": setups, "clock_start": cfg["run"]["clock_start"],
            "models": cfg["llm"]["models"], "provider": cfg["llm"]["provider"]}


@app.get("/customers")
def customers() -> list[dict[str, Any]]:
    """The accounts a client can sign in as (the demo page's account picker)."""
    try:
        return PostgresReader().customers()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def signed_in(customer_id: str | None, reader: PostgresReader | None = None) -> str | None:
    """The request's account, once it is known to exist; None when no one is signed in."""
    if customer_id and (reader or PostgresReader()).read("customers", customer_id).get("not_found"):
        raise ValueError(f"unknown customer account {customer_id}")
    return customer_id


def _run_chat(request: ChatRequest, tracer: Tracer, cancellation: CancellationToken | None = None) -> ChatResponse:
    """One turn; shared by /chat and /chat/stream."""
    cancellation = cancellation or CancellationToken()
    cancellation.check()
    customer_id = signed_in(request.customer_id)
    cfg = load_config(setup=request.setup)
    clock = Clock(request.clock_start or datetime.fromisoformat(cfg["run"]["clock_start"]))
    history = [turn.model_dump() for turn in request.history]
    output = run_turn(build_setup(cfg=cfg), _llm(), ticket_id=tracer.ticket_id, message=request.customer_message,
                      history=history, customer_id=customer_id, clock=clock, tracer=tracer,
                      cancellation=cancellation)
    return ChatResponse(
        ticket_id=output.ticket_id, setup=output.setup, strategy=output.strategy, customer_id=request.customer_id,
        reply=output.reply, ticket_status=output.status, note=output.note, handoff_packet=output.handoff_packet,
        evidence=output.evidence, writes=output.writes, parts=output.parts, reply_review=output.reply_review,
        audit_log=output.audit_log, tool_call_count=tracer.tool_call_count, tool_calls=tracer.tool_calls,
        trace=tracer.summary(),
        history=_retain_history([*history, {"role": "customer", "content": request.customer_message},
                                 *([{"role": "agent", "content": output.reply}] if output.reply else [])]),
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
    """The same turn as /chat as Server-Sent Events: a `trace` event per step as it is logged, then one
    `result` event holding the /chat response, or an `error` event."""
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
