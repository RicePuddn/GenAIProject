"""Prompt Guard screening, the first step of every turn in every setup.

Screen only the new customer message for this turn. Long messages are split into overlapping chunks that
fit the classifier's small context.
"""
from __future__ import annotations

from typing import Any

from SupportOpsBot.runtime import prompts
from SupportOpsBot.runtime.shared.context import TurnContext

CHUNK_CHARS = 1000
CHUNK_STEP = 800     # consecutive chunks overlap by 200 characters
OLLAMA_GUARD_PROMPT = prompts.load("guard")


def screen(ctx: TurnContext, message: str) -> str:
    """Return benign, malicious or error for this message; the first non-benign chunk decides."""
    ctx.log("node_start", node="guard")
    if not ctx.cfg["guard"]["enabled"]:
        return "benign"
    for offset in range(0, len(message), CHUNK_STEP):
        flag = classify(ctx, message[offset:offset + CHUNK_CHARS], role="customer", offset=offset)
        if flag != "benign":
            return flag
        if offset + CHUNK_CHARS >= len(message):
            break
    return "benign"


def classify(ctx: TurnContext, text: str, **source: Any) -> str:
    """One chunk: benign, malicious or error. Fails closed: anything unparseable is an error.
    `source` (turn index, role, offset) is recorded on the guard events."""
    messages = [{"role": "user", "content": text}]
    messages.insert(0, {"role": "system", "content": OLLAMA_GUARD_PROMPT})
    label = (ctx.chat("guard", messages).content or "").strip().lower()
    try:
        score = float(label)
    except ValueError:
        score = None
    if score is not None:
        if not 0 <= score <= 1:
            ctx.log("guard_unparseable", response=label, reason="score_out_of_range", **source)
            return "error"
        threshold = float(ctx.cfg["guard"]["threshold"])
        flagged = score >= threshold
        ctx.log("guard_classification", label="malicious" if flagged else "benign",
                malicious_score=score, threshold=threshold, **source)
        return "malicious" if flagged else "benign"
    flagged = label in {"malicious", "unsafe", "flagged"}
    if not flagged and label not in {"benign", "safe"}:
        ctx.log("guard_unparseable", response=label, reason="empty_or_unknown_label", **source)
        return "error"
    ctx.log("guard_classification", label="malicious" if flagged else "benign", **source)
    return "malicious" if flagged else "benign"
