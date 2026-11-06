"""What a human operator receives when a turn ends awaiting_human, and what the customer is told meanwhile or when a turn is cut off."""
from __future__ import annotations

from typing import Any
from SupportOpsBot.runtime.shared.tool_catalog import WRITE_ACTIONS, WRITE_REVIEW_TOOLS, operation_for_tool

HOLD_REPLY = "I’ve referred your request to a support specialist for review."
# The LLM provider or a tool failed, so the turn's outcome cannot be reported. It claims nothing about the
# request, or that the ticket was saved: the database may be what failed.
INTERRUPTED_REPLY = ("Sorry, we hit a technical problem and couldn’t finish your request. Please send your message "
                     "again in a little while.")


def handoff_packet(*, ticket_id: str, reason: str, message: str, evidence: list[dict[str, Any]],
                   note: str | None) -> dict[str, Any]:
    """The reason, the customer's request, every tool call and write in this turn, the clauses the writes
    cited, and the agent's own note, the same for every setup."""
    writes = []
    for item in evidence:
        if item["tool"] not in WRITE_ACTIONS:
            continue
        result, arguments = item["result"], item.get("arguments") or {}
        writes.append({
            "status": result.get("status"),
            "operation": result.get("operation") or operation_for_tool(item["tool"], arguments),
            "order_id": result.get("order_id") or arguments.get("order_id"),
            "amount": result.get("amount", arguments.get("amount")),
            "items": result.get("items", arguments.get("items", [])),
            "write_id": result.get("write_id"),
            "reason": result.get("reason") or arguments.get("reason"),
        })
    cited = [clause for item in evidence
             if item["tool"] in WRITE_REVIEW_TOOLS and isinstance(item["arguments"], dict)
             for clause in item["arguments"].get("cited_clauses", [])]
    return {
        "ticket_id": ticket_id, "reason": reason, "summary": message[:500],
        "actions_taken": [f"{item['agent']}: {item['tool']}" + (" (error)" if "error" in item["result"] else "")
                          for item in evidence],
        "writes": writes, "cited_clauses": list(dict.fromkeys(cited)),
        "recommendation": note or "Review the request; the automated flow stopped before an outcome.",
    }
