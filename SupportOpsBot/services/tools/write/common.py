"""Shared execution mechanics for the three named customer write tools."""
from __future__ import annotations

from typing import Any, Literal

from SupportOpsBot.services.structure.structure import BackendWriteArguments


WriteOperation = Literal["return_request", "refund", "cancel"]


def perform_write(registry: Any, operation: WriteOperation, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate a named write, enforce its gate/review, then use the idempotent executor."""
    payload = BackendWriteArguments.model_validate({**arguments, "operation": operation}).model_dump(exclude_none=True)

    # Check idempotency before the current database state: a prior cancellation changes that state.
    checked = registry.writes.prepare(payload)
    if checked.get("status") == "executed":
        return checked
    if checked["status"] != "allowed":
        return checked

    decision = checked["policy_check"]
    if decision["route"] == "approver" and not registry.gate.is_approved(payload):
        return {
            "status": "review_required",
            "policy_check": decision,
            "reason": "This write needs an independent review first: call request_review with the same "
                      "operation, order, amount and items and a review_reason, then call this write again "
                      "with the approved values.",
        }
    return registry.writes.apply(payload, decision)
