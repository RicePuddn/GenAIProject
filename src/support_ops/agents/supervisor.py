"""Supervisor: triages each message and drafts every customer reply.

Routing, approval and execution decisions are made by the graph and the gateway from the
triage and the specialist's proposal; the supervisor's LLM calls only classify and write.
"""
from __future__ import annotations

import json
from typing import Any

from support_ops.agents.models import Triage
from support_ops.agents.runtime import AgentContext, chat_json

TRIAGE_PROMPT = (
    "You triage customer support messages for an online store. Specialists: orders_shipping "
    "(order status, tracking, delivery delays, order changes such as cancellation or address change) "
    "and returns_refunds (return eligibility, return labels, exchanges, refunds). A request for money "
    "back or to send an item back goes to returns_refunds, even when the order is also late. Use specialist null "
    "when neither fits. Set wants_human when the customer asks for a person. Set duplicate_of only "
    "when the conversation shows this repeats an open ticket ID. Return JSON only, matching: "
)

REPLY_PROMPT = (
    "You write the reply to the customer for an online store's support team. Use only the outcome "
    "supplied: its status, the request, the facts verified with tools, the specialist's finding, and "
    "tell_customer, which says what to convey; follow it. Policy clauses among the facts govern what "
    "you may share (for example, which shipment details); follow them, and share only what answers "
    "the customer. State amounts, dates, statuses and products exactly as they appear, "
    "and mention no channel, email, link, number or date that is not in it. Write in customer terms: "
    "name products, never internal item IDs, policy clause IDs, rule names, thresholds, routes or "
    "internal review steps (such as an approver or auto-approval). Never say something is pending "
    "review unless the status is awaiting_human. Never say a refund or change happened unless "
    "tell_customer says it is recorded; the system is a simulation, so describe it as recorded, not "
    "as a payment sent. Use a neutral greeting, never an internal ID as a name. Be concise."
)


class Supervisor:
    def __init__(self, ctx: AgentContext):
        self.ctx = ctx

    def triage(self, brief: dict[str, Any]) -> Triage:
        self.ctx.log("node_start", node="triage")
        triage = chat_json(self.ctx, "supervisor", [
            {"role": "system", "content": TRIAGE_PROMPT + json.dumps(Triage.model_json_schema())},
            {"role": "user", "content": json.dumps(brief, default=str)},
        ], Triage)
        self.ctx.log("triage", **triage.model_dump(mode="json"))
        return triage

    def draft_reply(self, brief: dict[str, Any], outcome: dict[str, Any], issues: list[str] | None = None) -> str:
        self.ctx.log("node_start", node="draft_reply", status=outcome.get("status"))
        self.ctx.budget.check()
        payload = {**brief, "outcome": outcome}
        if issues:
            payload["fix_these_issues_from_review"] = issues
        reply = self.ctx.chat("supervisor", [
            {"role": "system", "content": REPLY_PROMPT +
             f" Current workflow date: {self.ctx.clock.now().date()}."},
            {"role": "user", "content": json.dumps(payload, default=str)},
        ])
        return (reply.content or "").strip()
