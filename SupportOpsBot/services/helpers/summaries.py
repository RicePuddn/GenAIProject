"""Small operational summaries over the full run trajectory."""
from __future__ import annotations

from collections import Counter
from typing import Any


def operational_summary(response: dict[str, Any]) -> dict[str, Any]:
    trace = response.get("trace") or {}
    events = trace.get("events") or []
    return {
        "setup": response.get("setup"),
        "ticket_status": response.get("ticket_status"),
        "llm_calls": trace.get("llm_calls", 0),
        "tool_call_count": trace.get("tool_call_count", response.get("tool_call_count", 0)),
        "evidence_count": len(response.get("evidence") or []),
        "write_count": len(response.get("writes") or []),
        "part_count": len(response.get("parts") or []),
        "handoff": bool(response.get("handoff_packet")),
        "cost_usd": trace.get("cost_usd", 0),
        "elapsed_s": trace.get("elapsed_s", 0),
        "event_counts": dict(sorted(Counter(event.get("kind", "unknown") for event in events).items())),
    }

