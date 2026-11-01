"""The customer reply, written and checked the same way in every setup, after the setup's body.

The reply writer drafts from the ticket (its status, the agent's note and the tool evidence). Code checks
the wording first, for rules an LLM reviewer can let through: a pending refund described as paid, and
internal IDs. A draft that fails is redrafted without spending a reply check, up to
review.max_wording_rewrites times. Then the approver checks the reply; a rejected draft is redrafted
with its issues, up to review.max_reply_drafts drafts. If none passes, the caller sends the hold reply.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from support_ops import prompts
from support_ops.shared.context import TurnContext
from support_ops.shared.gates.approver import ApprovalResult, Approver
from support_ops.shared.rules import AWAITING_STORE_RETURN_STATUSES

# "has been refunded", "refund of SGD 20.00 has been issued", "we have issued your refund", "the money has
# been returned", and the like.
PAID_CLAIM = re.compile(
    r"\b(?:has|have|was|were)\s+(?:already\s+)?(?:been\s+)?(?:refunded|credited|reimbursed|paid\s+back)\b"
    r"|\brefunds?\b(?:[^.!?]|\.(?=\d)){0,40}?\b(?:has|have|was|were)\s+(?:already\s+)?(?:been\s+)?"
    r"(?:processed|issued|completed|paid|sent|credited)\b"
    r"|\b(?:issued|processed|completed|paid|sent)\s+(?:you\s+)?(?:a|the|your)\s+(?:full\s+)?refund\b"
    r"|\b(?:money|payment|funds|amount)\s+(?:has|have|was|were)\s+(?:already\s+)?(?:been\s+)?"
    r"(?:returned|refunded|credited|sent\s+back|paid\s+back)\b",
    re.IGNORECASE,
)
# Policy clause IDs, e.g. ODR-OrderChangeCancel-4.2, REF-Refunds-6.6 (RAG and skills use the same IDs).
CLAUSE_ID = re.compile(r"\b(?:ODR|REF|RET|SHP)-[A-Z][A-Za-z]+-\d+(?:\.\d+)*\b")
# Record IDs other than the order number: items, payments, returns, refunds, writes, shipments, tickets,
# customers, products.
RECORD_ID = re.compile(r"\b(?:ITM|PAY|RET|RFD|CHG|EXC|SHP|SEV|TKT|CUS|PRD)-[A-Z0-9]+(?:-[A-Z0-9]+)*\b")
# Models often write IDs with non-breaking or other dashes (ORD‑2026‑00109).
DASHES = re.compile("[‐-―−]")


def refunds_in(evidence: list[dict[str, Any]]) -> tuple[bool, bool]:
    """(a refund is pending, a refund was processed), from the refunds this ticket wrote or looked up."""
    pending = processed = False
    for item in evidence:
        result = item.get("result") or {}
        if item.get("tool") == "backend_write" and result.get("operation") == "refund" \
                and result.get("status") == "executed":
            pending = True   # a recorded refund is pending: no money moves
        for refund in result.get("refunds", []) if isinstance(result.get("refunds"), list) else []:
            pending |= refund.get("status") == "pending"
            processed |= refund.get("status") == "processed"
    return pending, processed


def refund_wording_issue(reply: str, evidence: list[dict[str, Any]]) -> str | None:
    """Why the reply must be redrafted, or None. It may not say a pending refund has been paid."""
    pending, processed = refunds_in(evidence)
    if not pending or processed or not (claim := PAID_CLAIM.search(reply)):
        return None
    return (f'the reply says "{claim.group(0)}", but the refund is only recorded as pending and no money has '
            "moved: describe it as recorded and pending")


def internal_id_issue(reply: str) -> str | None:
    """Why the reply must be redrafted, or None. It may name the order, but no internal ID."""
    text = DASHES.sub("-", reply)
    found = sorted({match.group(0) for pattern in (CLAUSE_ID, RECORD_ID) for match in pattern.finditer(text)})
    if not found:
        return None
    return (f"the reply shows internal IDs {found}: remove them and describe the policy or record in plain "
            "words (the order number may stay)")


def wording_issues(reply: str, evidence: list[dict[str, Any]]) -> list[str]:
    """Every wording rule the reply breaks; empty when it may go on to the reply check."""
    return [issue for issue in (refund_wording_issue(reply, evidence), internal_id_issue(reply)) if issue]


def problems(evidence: list[dict[str, Any]], today: date) -> list[str]:
    """What has gone wrong on the store's side, found by code in the records the agents looked up this turn,
    so the apology does not depend on a model noticing. Only lookups count: a setup whose agents never
    looked at the shipments or returns has nothing listed."""
    found: list[str] = []
    for item in evidence:
        result = item["result"]
        if "error" in result:
            continue
        if item["tool"] == "get_shipments_for_order":
            for shipment in result.get("shipments", []):
                estimated = shipment.get("estimated_delivery")
                if estimated and not shipment.get("delivered_at") and shipment.get("status") != "cancelled"                         and _day(estimated) < today:
                    found.append(f"A shipment is past its estimated delivery date ({_day(estimated)}).")
        if item["tool"] == "get_returns_for_order":
            for request in result.get("returns", []):
                if request.get("status") in AWAITING_STORE_RETURN_STATUSES:
                    requested = _day(request["requested_at"]) if request.get("requested_at") else None
                    found.append("A return the customer requested" + (f" on {requested}" if requested else "")
                                 + " has not been processed by the store yet.")
    return list(dict.fromkeys(found))


def _day(value: Any) -> date:
    return value.date() if isinstance(value, datetime) else value if isinstance(value, date)         else date.fromisoformat(str(value)[:10])


def write_reply(ctx: TurnContext, approver: Approver, brief: dict[str, Any],
                ticket: dict[str, Any]) -> tuple[str | None, ApprovalResult | None]:
    """The approved reply and its review, or (None, the last review) when no draft passed."""
    ctx.log("node_start", node="reply")
    settings = ctx.cfg["review"]
    drafts, rewrites = settings["max_reply_drafts"], settings["max_wording_rewrites"]
    issues: list[str] | None = None
    review: ApprovalResult | None = None
    while drafts > 0:
        ctx.budget.gate_call()
        result = ctx.chat("reply_writer", [
            {"role": "system", "content": prompts.load("reply_writer", today=ctx.clock.now().date())},
            {"role": "user", "content": json.dumps({
                **brief, "ticket": ticket, **({"previous_draft_issues": issues} if issues else {})}, default=str)},
        ])
        reply = (result.content or "").strip()
        if not reply:
            issues, drafts = ["empty reply"], drafts - 1
            continue
        if wording := wording_issues(reply, ticket["evidence"]):
            ctx.log("reply_wording_rejected", issues=wording)
            issues = wording
            if rewrites > 0:
                rewrites -= 1
            else:
                drafts -= 1
            continue
        drafts -= 1
        review = approver.check_reply(brief, ticket, reply)
        if review.decision == "approve":
            ctx.log("customer_reply", reply=reply)
            return reply, review
        issues = review.issues or [review.reason]
    return None, review
