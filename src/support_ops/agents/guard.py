"""Prompt Guard screening, mirroring the single-agent `_guard` node so S1 stays untouched."""
from __future__ import annotations

from typing import Any, Callable

from support_ops.helpers.llm import ChatResult

Chat = Callable[..., ChatResult]
Log = Callable[..., None]


def classify(chat: Chat, cfg: dict[str, Any], message: str, log: Log) -> str:
    """Return benign, malicious or error. Fails closed: anything unparseable is an error."""
    guard = cfg.get("guard", {})
    if not guard.get("enabled", True):
        return "benign"
    # Groq's classifier requires exactly one user message.
    label = (chat("guard", [{"role": "user", "content": message}]).content or "").strip().lower()
    try:
        score = float(label)
    except ValueError:
        score = None
    if score is not None:
        if not 0 <= score <= 1:
            log("guard_unparseable", response=label, reason="score_out_of_range")
            return "error"
        threshold = float(guard.get("threshold", 0.5))
        flagged = score >= threshold
        log("guard_classification", label="malicious" if flagged else "benign",
            malicious_score=score, threshold=threshold)
        return "malicious" if flagged else "benign"
    flagged = label in {"malicious", "unsafe", "flagged"}
    if not flagged and label not in {"benign", "safe"}:
        log("guard_unparseable", response=label, reason="empty_or_unknown_label")
        return "error"
    log("guard_classification", label="malicious" if flagged else "benign")
    return "malicious" if flagged else "benign"
