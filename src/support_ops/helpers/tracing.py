"""Per-ticket trace: every LLM call, tool call, policy read and gateway event.

Traces are the raw data behind every reported number, so each metric in the report
can be traced back to the runs that produced it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Tracer:
    ticket_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    _t0: float = field(default_factory=time.perf_counter)

    def log(self, kind: str, **payload: Any) -> None:
        self.events.append({"t": round(time.perf_counter() - self._t0, 4), "kind": kind, **payload})

    def record_llm(self, role: str, model: str, input_tokens: int, output_tokens: int,
                   cost: float, latency_s: float, cached: bool) -> None:
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost
        self.log("llm_call", role=role, model=model, input_tokens=input_tokens,
                 output_tokens=output_tokens, cost_usd=round(cost, 6),
                 latency_s=round(latency_s, 3), cached=cached)

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["kind"] == kind]

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def tool_call_count(self) -> int:
        return len(self.of_kind("tool_start"))

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        completed = {event["call_id"]: event for event in self.of_kind("tool_end")}
        return [
            {**event, "execution": completed.get(event["call_id"])}
            for event in self.of_kind("tool_call_requested")
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "elapsed_s": round(self.elapsed_s, 4),
            "tool_call_count": self.tool_call_count,
            "events": self.events,
        }

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "ticket_id": self.ticket_id, **self.summary(),
        }, indent=1, default=str))
