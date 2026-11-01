"""`run_turn`: guard → setup body → reply and reply check → ticket save, the same for every setup.

If the LLM provider fails, or a tool call fails for good and the turn would still end resolved, it ends
`in_progress`: the ticket is saved and the customer is asked to send their message again. Sending it on the
same ticket is safe, because each write replays by its key.

The pipeline builds the turn's shared services (the account-scoped database, the policy provider, the
rules, the write gate and the tool registry) and runs the gates around the setup's body, so a setup
implements only its body and never calls the guard, the reply check or the ticket save itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from support_ops.setups.base import Setup, Turn, TurnResult
from support_ops.shared.budget import Budget, BudgetExceeded, Retries
from support_ops.shared.cancellation import CancellationToken
from support_ops.shared.clock import Clock
from support_ops.shared.context import TurnContext
from support_ops.shared.db import PostgresReader, PostgresWriter
from support_ops.shared.db.accounts import for_account
from support_ops.shared.gates.approver import Approver
from support_ops.shared.gates.guard import screen
from support_ops.shared.gates.handoff import HOLD_REPLY, INTERRUPTED_REPLY, handoff_packet
from support_ops.shared.gates.reply import problems, write_reply
from support_ops.shared.gates.ticket import TicketLog, save_ticket
from support_ops.shared.gates.write_gate import WriteExecutor, WriteGate
from support_ops.shared.llm import LLM, ProviderError
from support_ops.shared.policy.provider import PolicyProvider, policy_from_config
from support_ops.shared.rules import BackendFacts, Rules
from support_ops.shared.tools import ToolRegistry
from support_ops.shared.tracing import Tracer

# Both roles are untrusted client input, never authenticated assistant messages.
HISTORY_ROLES = {"customer", "agent"}
DASHES = str.maketrans({dash: "-" for dash in "‐‑‒–—―−﹣－"})


def plain_dashes(text: str) -> str:
    """Unicode dashes (‑ – —) as "-". Models misread them inside IDs: ORD‑00042 was looked up as ORD-100042."""
    return text.translate(DASHES)


@dataclass
class TurnOutput:
    """One turn's outcome, the same shape for every setup."""
    ticket_id: str
    setup: str
    strategy: dict[str, str]
    reply: str
    status: str
    note: str | None
    handoff_packet: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    writes: list[dict[str, Any]]
    parts: list[dict[str, Any]]
    reply_review: dict[str, Any] | None
    audit_log: list[dict[str, Any]]
    trace: Tracer
    history: list[dict[str, str]] = field(default_factory=list)


def interrupted(ctx: TurnContext, exc: ProviderError, parts: list[dict[str, Any]] | None = None) -> TurnResult:
    """The provider failed, so the customer has not heard the outcome: the ticket stays in progress."""
    ctx.log("provider_error", error=str(exc.__cause__ or exc)[:300])
    return TurnResult("in_progress", f"provider unavailable: {exc}", hold=True, parts=parts or [])


def run_turn(setup: Setup, llm: LLM, *, ticket_id: str, message: str, history: list[dict[str, str]] | None = None,
             customer_id: str | None = None, reader: PostgresReader | None = None,
             writer: PostgresWriter | None = None, policy: PolicyProvider | None = None, clock: Clock | None = None,
             tracer: Tracer | None = None, cancellation: CancellationToken | None = None) -> TurnOutput:
    """Run one turn of `setup`. `history` holds the ticket's earlier turns as
    {"role": "customer" | "agent", "content": ...}; nothing is kept between calls."""
    cfg = setup.cfg
    history = [{"role": turn["role"], "content": turn["content"]} for turn in history or []]
    if any(turn["role"] not in HISTORY_ROLES for turn in history):
        raise ValueError("history roles must be customer or agent")
    tracer = tracer or Tracer(ticket_id)
    clock = clock or Clock(datetime.fromisoformat(cfg["run"]["clock_start"]))
    ctx = TurnContext(cfg, ticket_id, llm, clock, tracer, Budget(cfg["limits"], tracer), Retries.from_config(cfg),
                      cancellation or CancellationToken())
    # The signed-in customer: every read and write is scoped to their account, and the ticket is theirs.
    reader, writer = for_account(reader or PostgresReader(), writer or PostgresWriter(), customer_id)
    rules = Rules.load(cfg["policy"]["rules_file"])
    policy = policy or policy_from_config(cfg, rules)
    brief: dict[str, Any] = {"message": plain_dashes(message)}
    if history:
        brief["history"] = [{**turn, "content": plain_dashes(turn["content"])} for turn in history]
    if customer_id:
        brief["signed_in_account"] = customer_id
    log = TicketLog(ticket_id, clock)
    approver = Approver(ctx)
    gate = WriteGate(ctx, BackendFacts(reader, clock), rules, policy, log,
                     reviewer=lambda request: approver.check_write(brief, request))
    executor = WriteExecutor(ctx, writer, log)

    turn = Turn(ctx, brief, ToolRegistry(reader, policy, gate, executor), gate, executor, policy, customer_id)

    ctx.log("workflow_start", setup=cfg["setup"]["id"], strategy=cfg["strategy"], history_turns=len(history))
    try:
        flag = screen(ctx, message)
        if flag != "benign":
            result = TurnResult("awaiting_human", f"guard: {flag}", hold=True)
        else:
            result = setup.run(turn)
    except BudgetExceeded as exc:
        ctx.log("budget_exceeded", reason=str(exc))
        result = TurnResult("awaiting_human", f"budget: {exc}", hold=True, parts=[])
    except ProviderError as exc:
        result = interrupted(ctx, exc)

    if result.status == "resolved" and ctx.failed_calls:
        # An agent cannot call a ticket resolved over work that never ran.
        ctx.log("tool_failure_unfinished", failed=ctx.failed_calls)
        result = TurnResult("in_progress", f"a tool failed: {next(iter(ctx.failed_calls.values()))[:200]}", hold=True,
                            parts=result.parts)

    reply, review = HOLD_REPLY, None
    if not result.hold:
        ticket = {"status": result.status, "note": result.note, "evidence": ctx.evidence,
                  "problems": problems(ctx.evidence, clock.now().date())}
        try:
            drafted, review = write_reply(ctx, approver, brief, ticket)
        except BudgetExceeded as exc:
            ctx.log("budget_exceeded", reason=str(exc))
            drafted = None
        except ProviderError as exc:
            drafted = None
            if result.status != "awaiting_human":   # a person already has it otherwise; the hold reply stands
                result = interrupted(ctx, exc, result.parts)
        if drafted is not None:
            reply = drafted
        elif result.status != "in_progress":
            result = replace(result, status="awaiting_human",
                             note=result.note if result.status == "awaiting_human" else "reply failed review")
    if result.status == "in_progress":
        reply = INTERRUPTED_REPLY

    packet = None
    if result.status == "awaiting_human":
        packet = handoff_packet(ticket_id=ticket_id, reason=result.note or "review required", message=message,
                                evidence=ctx.evidence, note=None if result.hold else result.note)
        log.audit("handoff", reason=packet["reason"])
        ctx.log("handoff_created", packet=packet)
    first_message = next((turn["content"] for turn in history if turn["role"] == "customer"), message)
    save_ticket(ctx, writer, first_message=first_message, status=result.status, reply=reply, handoff_packet=packet)
    ctx.log("workflow_end", status=result.status, llm_calls=tracer.llm_calls, elapsed_s=round(tracer.elapsed_s, 4))
    return TurnOutput(
        ticket_id=ticket_id, setup=cfg["setup"]["id"], strategy=dict(cfg["strategy"]), reply=reply,
        status=result.status, note=result.note, handoff_packet=packet, evidence=ctx.evidence, writes=ctx.writes,
        parts=result.parts, reply_review=review.model_dump(mode="json") if review else None,
        audit_log=log.events, trace=tracer, history=history)
