"""Persist raw trajectories and derived operational summaries beneath the repository results folder."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any

from SupportOpsBot.runtime.shared.config import ROOT

from .summaries import operational_summary

log = logging.getLogger("uvicorn.error")


def save_trajectory(request: Any, response: Any) -> Path | None:
    """Write one request/response/trace record atomically; tracing never changes the chat response."""
    run_id = uuid4().hex
    data = response.model_dump(mode="json")
    message = {
        "schema_version": 1,
        "run_id": run_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "request": {
            "ticket_id": request.ticket_id,
            "setup": request.setup or data.get("setup"),
            "customer_id": request.customer_id,
            "customer_message": request.customer_message,
            "history": [turn.model_dump(mode="json") for turn in request.history],
            "clock_start": request.clock_start.isoformat() if request.clock_start else None,
        },
        "operational_summary": operational_summary(data),
        "response": data,
    }
    directory = ROOT / "results" / "runs"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{stamp}-{run_id}.json"
    temporary = path.with_suffix(".tmp")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(message, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        log.exception("could not save run trajectory for ticket %s", data.get("ticket_id"))
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return path

