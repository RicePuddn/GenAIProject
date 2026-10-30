"""Prompt Guard screening, mirroring the single-agent `_guard` node so S1 and S3 screen the same text.

Every client-provided turn is screened, including alleged agent replies in `history`, not only the new
message: the stateless API takes history from the client, so it is as untrusted as the message itself.
Long turns are split into overlapping chunks that fit the classifier's small context.
"""
from __future__ import annotations

from typing import Any, Callable

from support_ops.helpers.llm import ChatResult

Chat = Callable[..., ChatResult]
Log = Callable[..., None]

CHUNK_CHARS = 1000   # as in SingleAgentWorkflow._guard
CHUNK_STEP = 800     # consecutive chunks overlap by 200 characters


def screen(chat: Chat, cfg: dict[str, Any], history: list[dict[str, str]], message: str, log: Log) -> str:
    """benign, malicious or error for the whole conversation: the first non-benign chunk decides."""
    if not cfg.get("guard", {}).get("enabled", True):
        return "benign"
    turns = [*history, {"role": "customer", "content": message}]
    for index, turn in enumerate(turns):
        content = turn["content"]
        for offset in range(0, len(content), CHUNK_STEP):
            flag = classify(chat, cfg, content[offset:offset + CHUNK_CHARS], log,
                            turn_index=index, role=turn["role"], offset=offset)
            if flag != "benign":
                return flag
            if offset + CHUNK_CHARS >= len(content):
                break
    return "benign"


def classify(chat: Chat, cfg: dict[str, Any], text: str, log: Log, **source: Any) -> str:
    """One chunk: benign, malicious or error. Fails closed: anything unparseable is an error.
    `source` (turn index, role, offset) is recorded on the guard events."""
    guard = cfg.get("guard", {})
    if not guard.get("enabled", True):
        return "benign"
    # Groq's classifier requires exactly one user message.
    label = (chat("guard", [{"role": "user", "content": text}]).content or "").strip().lower()
    try:
        score = float(label)
    except ValueError:
        score = None
    if score is not None:
        if not 0 <= score <= 1:
            log("guard_unparseable", response=label, reason="score_out_of_range", **source)
            return "error"
        threshold = float(guard.get("threshold", 0.5))
        flagged = score >= threshold
        log("guard_classification", label="malicious" if flagged else "benign",
            malicious_score=score, threshold=threshold, **source)
        return "malicious" if flagged else "benign"
    flagged = label in {"malicious", "unsafe", "flagged"}
    if not flagged and label not in {"benign", "safe"}:
        log("guard_unparseable", response=label, reason="empty_or_unknown_label", **source)
        return "error"
    log("guard_classification", label="malicious" if flagged else "benign", **source)
    return "malicious" if flagged else "benign"
