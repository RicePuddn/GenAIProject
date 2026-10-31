"""Deterministic checks on a customer reply's wording, for rules an LLM reviewer can let through.

No money moves: an agent's refund is recorded as pending, so a reply that says the money has been
refunded is false however the rest of it reads. Internal IDs (policy clauses, items, payments, writes,
tickets) are never for the customer. Both are caught by pattern, not by judgment.
"""
from __future__ import annotations

import re
from typing import Any

# "has been refunded", "refund has been processed", "the money has been returned", and the like.
PAID_CLAIM = re.compile(
    r"\b(?:has|have|was|were)\s+(?:already\s+)?(?:been\s+)?(?:refunded|credited|reimbursed|paid\s+back)\b"
    r"|\brefund\s+(?:has|was)\s+(?:already\s+)?(?:been\s+)?(?:processed|issued|completed|paid|sent|credited)\b"
    r"|\b(?:money|payment|funds|amount)\s+(?:has|have|was|were)\s+(?:already\s+)?(?:been\s+)?"
    r"(?:returned|refunded|credited|sent\s+back|paid\s+back)\b",
    re.IGNORECASE,
)
# Policy clause IDs, e.g. ODR-OrderChangeCancel-4.2, REF-Refunds-6.6 (RAG and skills use the same IDs).
CLAUSE_ID = re.compile(r"\b(?:ODR|REF|RET|SHP)-[A-Z][A-Za-z]+-\d+(?:\.\d+)*\b")
# Record IDs other than the order number the customer gave: items, payments, returns, refunds, writes,
# shipments, tickets, customers, products.
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
