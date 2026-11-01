"""Test doubles: a scripted LLM, an in-memory database and a fake retriever. With them a whole turn of any
setup runs with no API key, no PostgreSQL and no Pinecone."""
from __future__ import annotations

import copy
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from support_ops.shared.budget import Budget, Retries
from support_ops.shared.clock import Clock
from support_ops.shared.config import load_config
from support_ops.shared.context import TurnContext
from support_ops.shared.db import UNKNOWN_CUSTOMER, WriteRefused
from support_ops.shared.gates.ticket import TicketLog
from support_ops.shared.gates.write_gate import WriteExecutor, WriteGate
from support_ops.shared.llm import ChatResult, ToolCall, count_tokens
from support_ops.shared.policy.rag import RagPolicy
from support_ops.shared.policy.skills import SkillsPolicy
from support_ops.shared.rules import BackendFacts, Rules
from support_ops.shared.tools import ALL_TOOLS, ToolRegistry
from support_ops.shared.tracing import Tracer

NOW = datetime(2026, 10, 1, 9, tzinfo=timezone(timedelta(hours=8)))   # run.clock_start
ORDER = "ORD-2026-00001"     # delivered five days ago, with an approved return: refunds are allowed
OPEN_ORDER = "ORD-2026-00002"  # still processing: it can be cancelled, then refunded
CUSTOMER = "CUS-1"

# ----------------------------------------------------------------------------- LLM

Responder = Callable[[list[dict[str, Any]], list[dict] | None], ChatResult]


def tool_call(name: str, _id: str | None = None, **arguments: Any) -> ChatResult:
    """One tool call, as a model returns it."""
    return ChatResult(None, [ToolCall(_id or f"call_{name}", name, arguments)])


def text(content: str) -> ChatResult:
    return ChatResult(content)


def as_json(value: Any) -> ChatResult:
    return ChatResult(json.dumps(value))


def plan(*calls: tuple[str, dict[str, Any]]) -> ChatResult:
    """A plan_execute plan."""
    return as_json({"calls": [{"name": name, "arguments": arguments} for name, arguments in calls]})


APPROVE = as_json({"decision": "approve", "issues": [], "reason": "Supported by the evidence."})


def reject(issue: str) -> ChatResult:
    return as_json({"decision": "reject", "issues": [issue], "reason": "Not supported."})


def escalate(issue: str) -> ChatResult:
    return as_json({"decision": "escalate", "issues": [issue], "reason": "A person must decide."})


class ScriptedLLM:
    """Per role, a queue of results, then a default responder. By default the guard finds every turn
    benign, the approver approves and the reply writer answers, so a test scripts only the agents."""

    def __init__(self) -> None:
        self.queues: dict[str, deque[ChatResult]] = defaultdict(deque)
        self.responders: dict[str, Responder] = {
            "guard": lambda *_: text("0.01"),
            "approver": lambda *_: APPROVE,
            "reply_writer": lambda *_: text("Thanks for your message: here is what we found."),
        }
        self.calls: list[tuple[str, list[dict[str, Any]], list[dict] | None]] = []

    def push(self, role: str, *results: ChatResult) -> ScriptedLLM:
        self.queues[role].extend(results)
        return self

    def on(self, role: str, fn: Responder) -> ScriptedLLM:
        self.responders[role] = fn
        return self

    def chat(self, role, messages, tools=None, tracer=None, json_mode=False) -> ChatResult:
        self.calls.append((role, copy.deepcopy(messages), tools))
        if self.queues[role]:
            result = copy.copy(self.queues[role].popleft())
        elif role in self.responders:
            result = self.responders[role](messages, tools)
        else:
            raise AssertionError(f"ScriptedLLM has no response left for role {role!r}")
        result.input_tokens = result.input_tokens or count_tokens(json.dumps(messages, default=str))
        result.output_tokens = result.output_tokens or count_tokens(
            (result.content or "") + json.dumps([call.arguments for call in result.tool_calls]))
        if tracer is not None:
            tracer.record_llm(role, "scripted", result.input_tokens, result.output_tokens, 0.0, 0.0, False)
        return result

    def roles(self) -> list[str]:
        return [role for role, _, _ in self.calls]

    def user_payload(self, role: str, index: int = -1) -> dict[str, Any]:
        """The JSON user message of one of `role`'s calls (the last by default)."""
        messages = [messages for called, messages, _ in self.calls if called == role][index]
        return json.loads(next(message["content"] for message in messages if message["role"] == "user"))


# ----------------------------------------------------------------------------- database


def seed_tables() -> dict[str, dict[str, dict[str, Any]]]:
    """Two orders of one customer. ORDER: delivered five days ago, both lines returned and approved, so a
    refund up to what was paid is allowed. OPEN_ORDER: processing, nothing packed, so it can be cancelled."""
    delivered = NOW - timedelta(days=5)
    lines = [
        {"id": "ITM-1", "order_id": ORDER, "product_id": "PRD-1", "sku": "HP-1", "name": "Headphones",
         "category": "audio", "quantity": 1, "unit_price": "60.00", "paid_amount": "60.00",
         "item_status": "fulfilled", "final_sale": False, "product_type": "physical"},
        {"id": "ITM-2", "order_id": ORDER, "product_id": "PRD-2", "sku": "BT-1", "name": "Bottle",
         "category": "kitchen", "quantity": 2, "unit_price": "20.00", "paid_amount": "40.00",
         "item_status": "fulfilled", "final_sale": False, "product_type": "physical"},
        {"id": "ITM-3", "order_id": OPEN_ORDER, "product_id": "PRD-3", "sku": "LP-1", "name": "Lamp",
         "category": "home", "quantity": 1, "unit_price": "45.00", "paid_amount": "45.00",
         "item_status": "open", "final_sale": False, "product_type": "physical"},
    ]
    return {
        "customers": {CUSTOMER: {"id": CUSTOMER, "first_name": "Ana", "last_name": "Lim", "email": "ana@example.com",
                                 "loyalty_tier": "gold"},
                      "CUS-2": {"id": "CUS-2", "first_name": "Ben", "last_name": "Tan", "email": "ben@example.com",
                                "loyalty_tier": "none"}},
        "products": {},
        "orders": {
            ORDER: {"id": ORDER, "customer_id": CUSTOMER, "status": "delivered", "total": "100.00",
                    "currency": "SGD", "placed_at": NOW - timedelta(days=12)},
            OPEN_ORDER: {"id": OPEN_ORDER, "customer_id": CUSTOMER, "status": "processing", "total": "45.00",
                         "currency": "SGD", "placed_at": NOW - timedelta(days=1)},
        },
        "order_items": {line["id"]: line for line in lines},
        "payments": {"PAY-1": {"id": "PAY-1", "order_id": ORDER, "method": "card", "status": "captured",
                               "amount": "100.00"},
                     "PAY-2": {"id": "PAY-2", "order_id": OPEN_ORDER, "method": "card", "status": "captured",
                               "amount": "45.00"}},
        "shipments": {"SHP-1": {"id": "SHP-1", "order_id": ORDER, "status": "delivered", "tracking_number": "TRK1",
                                "shipped_at": NOW - timedelta(days=8), "delivered_at": delivered,
                                "estimated_delivery": (NOW - timedelta(days=6)).date(), "label_created_at": None,
                                "items": [{"shipment_id": "SHP-1", "order_item_id": "ITM-1", "quantity": 1},
                                          {"shipment_id": "SHP-1", "order_item_id": "ITM-2", "quantity": 2}]}},
        "shipment_events": {"SEV-1": {"id": "SEV-1", "shipment_id": "SHP-1", "tracking_number": "TRK1",
                                      "occurred_at": delivered, "status": "delivered", "location": "Singapore",
                                      "details": "Delivered"}},
        "returns": {"RET-1": {
            "id": "RET-1", "order_id": ORDER, "status": "approved", "reason": "faulty",
            "requested_at": NOW - timedelta(days=4), "received_at": NOW - timedelta(days=3),
            "approved_at": NOW - timedelta(days=2), "reviewed_at": NOW - timedelta(days=2),
            "evidence_status": "verified", "inspection_status": "passed", "dispute_status": "none",
            "resolution": "refund", "authorized_at": None, "handed_over_at": None, "extension_until": None,
            "extension_approved_at": None,
            "items": [{"return_id": "RET-1", "order_item_id": line["id"], "quantity": line["quantity"],
                       "unused": True, "original_packaging": True, "tags_complete": True,
                       "accessories_complete": True, "approved_refund_amount": line["paid_amount"]}
                      for line in lines[:2]]}},
        "refunds": {},
        "support_tickets": {},
    }


class FakeDatabase:
    """`PostgresReader` and `PostgresWriter` over in-memory tables. Writes change what later reads see,
    as in PostgreSQL: a cancelled order reads as cancelled, a recorded refund as pending."""

    def __init__(self, tables: dict[str, dict[str, dict[str, Any]]] | None = None):
        self.tables = tables if tables is not None else seed_tables()
        self.applied: dict[str, dict[str, Any]] = {}
        self.ticket_saves: list[dict[str, Any]] = []

    # reads ------------------------------------------------------------------

    def read(self, kind: str, record_id: str) -> dict[str, Any]:
        row = next((row for key, row in self.tables[kind].items() if key.lower() == record_id.lower()), None)
        return copy.deepcopy(row) if row is not None else {"not_found": True, "kind": kind, "id": record_id}

    def _for_order(self, table: str, order_id: str) -> list[dict[str, Any]]:
        return [copy.deepcopy(row) for row in self.tables[table].values() if row["order_id"].lower() == order_id.lower()]

    def items_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return self._for_order("order_items", order_id)

    def payments_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return self._for_order("payments", order_id)

    def shipments_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return self._for_order("shipments", order_id)

    def shipment_events_for_order(self, order_id: str) -> list[dict[str, Any]]:
        shipments = {row["id"] for row in self.shipments_for_order(order_id)}
        return [copy.deepcopy(row) for row in self.tables["shipment_events"].values() if row["shipment_id"] in shipments]

    def returns_for_order(self, order_id: str) -> dict[str, list[dict[str, Any]]]:
        return {"returns": self._for_order("returns", order_id), "refunds": self._for_order("refunds", order_id)}

    def orders_for_customer(self, customer_id: str) -> list[dict[str, Any]]:
        return [{"id": row["id"], "placed_at": row["placed_at"], "status": row["status"], "total": row["total"],
                 "currency": row["currency"], "items": ", ".join(item["name"] for item in self.items_for_order(row["id"]))}
                for row in self.tables["orders"].values() if row["customer_id"] == customer_id]

    def customers(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(row) for row in self.tables["customers"].values()]

    # writes -----------------------------------------------------------------

    def replay(self, idempotency_key: str) -> dict[str, Any] | None:
        return copy.deepcopy(self.applied.get(idempotency_key))

    def apply(self, request: dict[str, Any], *, idempotency_key: str, reason: str, at: datetime,
              payment_id: str | None = None, return_id: str | None = None,
              customer_id: str | None = None) -> dict[str, Any]:
        order = self.tables["orders"].get(request["order_id"])
        if order is None or (customer_id and order["customer_id"] != customer_id):
            raise WriteRefused(f"order {request['order_id']} not found")
        write_id = f"W-{len(self.applied) + 1:05d}"
        if request["operation"] == "refund":
            payment_id = payment_id or next(row["id"] for row in self.payments_for_order(order["id"])
                                            if row["status"] == "captured")
            self.tables["refunds"][write_id] = {"id": write_id, "order_id": order["id"], "payment_id": payment_id,
                                                "status": "pending", "amount": str(request["amount"]),
                                                "reason": reason, "requested_at": at, "return_id": return_id}
        elif request["operation"] == "order_change":
            if order["status"] != "processing":
                raise WriteRefused(f"order {order['id']} is {order['status']}, so it cannot be cancelled")
            order["status"] = "cancelled"
            for line in self.tables["order_items"].values():
                if line["order_id"] == order["id"]:
                    line["item_status"] = "cancelled"
        result = {"status": "executed", **request, "currency": "SGD", "payment_id": payment_id,
                  "write_id": write_id, "executed_at": at.isoformat()}
        self.applied[idempotency_key] = {"request": request, "result": result}
        return copy.deepcopy(result)

    def save_ticket(self, **ticket: Any) -> dict[str, Any]:
        self.ticket_saves.append(ticket)
        return {"id": ticket["ticket_id"], "status": ticket["status"],
                "customer_id": ticket.get("customer_id") or UNKNOWN_CUSTOMER, "order_id": ticket.get("order_id")}

    @property
    def writes(self) -> list[dict[str, Any]]:
        return [entry["result"] for entry in self.applied.values()]


# ----------------------------------------------------------------------------- retriever

CLAUSES = [
    {"id": "REF-Refunds-6.1", "title": "Refund method", "section": "Returns and refunds",
     "source": "returns-and-refunds-policy.pdf", "text": "Refunds go back to the original payment method.",
     "references": []},
    {"id": "ODR-OrderChangeCancel-4.2", "title": "Cancellation refunds", "section": "Orders and shipping",
     "source": "orders-and-shipping-policy.pdf", "text": "A cancelled order is refunded its captured balance.",
     "references": []},
]


class FakeRetriever:
    """Every search returns the same clauses, ranked."""

    def search(self, query: str) -> list[dict[str, Any]]:
        return [{**clause, "score": round(1 - index / 10, 2), "via": "search"} for index, clause in enumerate(CLAUSES)]

    def sections(self) -> list[str]:
        return ["Orders and shipping", "Returns and refunds"]

    def warm_up(self) -> None:
        pass


def clock() -> Clock:
    return Clock(NOW)


@dataclass
class Services:
    """One turn's shared services, without a setup: for testing the gates and tools directly."""
    ctx: TurnContext
    db: Any
    policy: Any
    gate: WriteGate
    executor: WriteExecutor
    registry: ToolRegistry
    log: TicketLog
    reviews: list[dict[str, Any]]

    def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        """One tool call by the agent, through the full tool path (budget, retries, evidence)."""
        return self.registry.toolset(ALL_TOOLS).call(self.ctx, "agent", self.ctx.next_call_id(), name, arguments)


def services(db: Any = None, *, mode: str = "skills", verdicts: list[str] | None = None,
             cfg: dict[str, Any] | None = None, llm: Any = None, writer: Any = None,
             ticket_id: str = "T1") -> Services:
    """`verdicts` are the reviewer's decisions in order (approve after they run out). `db` reads and, unless
    a `writer` is given, writes."""
    cfg = cfg or load_config(setup="s2-a" if mode == "skills" else "s1-a")
    db = db if db is not None else FakeDatabase()
    tracer, turn_clock = Tracer(ticket_id), clock()
    ctx = TurnContext(cfg, ticket_id, llm or ScriptedLLM(), turn_clock, tracer, Budget(cfg["limits"], tracer),
                  Retries.from_config(cfg))
    rules = Rules.load(cfg["policy"]["rules_file"])
    policy = SkillsPolicy(cfg["policy"]["skills_dir"], rules) if mode == "skills" else RagPolicy(FakeRetriever(), rules)
    log, reviews, pending = TicketLog(ticket_id, turn_clock), [], list(verdicts or [])

    def reviewer(request: dict[str, Any]) -> dict[str, Any]:
        reviews.append(request)
        decision = pending.pop(0) if pending else "approve"
        return {"decision": decision, "issues": [] if decision == "approve" else [f"{decision}ed by the test"],
                "reason": "test"}

    gate = WriteGate(ctx, BackendFacts(db, turn_clock), rules, policy, log, reviewer)
    executor = WriteExecutor(ctx, writer if writer is not None else db, log)
    return Services(ctx, db, policy, gate, executor, ToolRegistry(db, policy, gate, executor), log, reviews)
