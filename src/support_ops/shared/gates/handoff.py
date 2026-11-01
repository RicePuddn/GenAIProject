"""What a human operator receives when a turn ends awaiting_human, and what the customer is told meanwhile."""
from __future__ import annotations

from typing import Any

HOLD_REPLY = "I’ve referred your request to a support specialist for review."


def handoff_packet(*, ticket_id: str, reason: str, message: str, evidence: list[dict[str, Any]],
                   note: str | None) -> dict[str, Any]:
    """The reason, the customer's request, every tool call and write in this turn, the clauses the writes
    cited, and the agent's own note, the same for every setup."""
    writes = [{key: item["result"].get(key) for key in ("status", "operation", "order_id", "amount", "items",
                                                        "write_id", "reason")}
              for item in evidence if item["tool"] == "backend_write"]
    cited = [clause for item in evidence
             if item["tool"] in {"backend_write", "request_review"} and isinstance(item["arguments"], dict)
             for clause in item["arguments"].get("cited_clauses", [])]
    return {
        "ticket_id": ticket_id, "reason": reason, "summary": message[:500],
        "actions_taken": [f"{item['agent']}: {item['tool']}" + (" (error)" if "error" in item["result"] else "")
                          for item in evidence],
        "writes": writes, "cited_clauses": list(dict.fromkeys(cited)),
        "recommendation": note or "Review the request; the automated flow stopped before an outcome.",
    }
