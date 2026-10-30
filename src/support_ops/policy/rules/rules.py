"""Evaluate recorded facts against the grouped checks and outcomes in rules.json."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from support_ops.helpers.config import ROOT

DEFAULT_RULES_FILE = Path(__file__).with_name("rules.json")
OPERATIONS = ("refund", "return_request", "return_dispatch", "cancel", "delivery")
GROUPS = ("human_review", *OPERATIONS)


# -------------------- Input validation and conversion --------------------
# These helpers prepare values for comparison; they do not decide eligibility.

# Use Decimal for money and reject missing, boolean, negative or non-finite values.
def _money(value: Any) -> Decimal:
    try:
        if value is None or isinstance(value, bool):
            raise ValueError("amount is required")
        number = Decimal(str(value))
        if not number.is_finite() or number < 0:
            raise ValueError("amount must be finite and non-negative")
        return number
    except InvalidOperation as exc:
        raise ValueError("invalid monetary amount") from exc


# Convert an ISO date or timestamp to a calendar date in the policy timezone.
def _date(value: Any, timezone: str) -> date:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return value
    elif isinstance(value, str):
        if len(value) == 10:
            return date.fromisoformat(value)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("a recorded ISO date or timestamp is required")
    zone = ZoneInfo(timezone)
    return (parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed.astimezone(zone)).date()


# Reject malformed outcome data, including the contradictory true/true combination.
def _validate_outcome(outcome: dict) -> None:
    if any(type(outcome.get(key)) is not bool for key in ("allowed", "requires_human")):
        raise ValueError("rule outcomes need boolean allowed and requires_human values")
    if outcome["allowed"] and outcome["requires_human"]:
        raise ValueError("a rule cannot allow automatic processing and require human review together")
    if not isinstance(outcome.get("reason"), str) or not outcome["reason"].strip():
        raise ValueError("rule outcomes need a reason")


@dataclass(frozen=True)
class Rules:
    version: str
    currency: str
    refund_limit: Decimal
    timezone: str
    data: dict[str, Any]
    unsupported_operation: dict

    # -------------------- Load configuration and build results --------------------

    # Validate the JSON checks and their reviewed policy-PDF fingerprints.
    @classmethod
    def load(cls, path: str | Path = DEFAULT_RULES_FILE) -> Rules:
        from support_ops.policy.rag.identifiers import policy_source

        path = Path(path)
        try:
            config = json.loads((path if path.is_absolute() else ROOT / path).read_text())
            data = config["groups"]
            if set(data) != set(GROUPS):
                raise ValueError("rules.json must contain the six supported rule groups")
            limit = _money(data["refund"]["max_auto_amount"])
            if limit <= 0:
                raise ValueError("refund limit must be positive")
            ZoneInfo(config["timezone"])
            for group, field in (("return_request", "window_calendar_days"),
                                 ("return_dispatch", "window_calendar_days"),
                                 ("delivery", "investigate_after_calendar_days")):
                if type(data[group][field]) is not int or data[group][field] < 1:
                    raise ValueError(f"{group}.{field} must be a positive integer")
            for group in GROUPS:
                if not data[group]["rule_id"] or not data[group]["sources"] or not data[group]["checks"]:
                    raise ValueError(f"{group} needs a rule ID, sources and checks")
                for check in data[group]["checks"].values():
                    if not check["given"]:
                        raise ValueError(f"{group} has a check without a condition description")
                    _validate_outcome(check["outcome"])
            _validate_outcome(config["unsupported_operation"])
            sources = {policy_source(ref) for group in GROUPS for ref in data[group]["sources"]}
            if sources != set(config["policy_documents"]):
                raise ValueError("policy fingerprints must cover every referenced PDF")
            for name, expected in config["policy_documents"].items():
                source = ROOT / "policy/raw_policy/pdf" / name
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"{name} changed: review its hard rules before updating the fingerprint")
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"invalid rules file {path}: {exc!r}") from exc
        return cls(
            version=str(config["version"]),
            currency=config["currency"],
            refund_limit=limit,
            timezone=config["timezone"],
            data=data,
            unsupported_operation=config["unsupported_operation"],
        )

    # Copy the JSON outcome, then add the rule identity, policy sources and case details.
    def _result(self, group: str, check: str, **details) -> dict:
        definition = self.data[group]
        outcome = definition["checks"][check]["outcome"]

        result = dict(outcome)
        result["rule"] = definition["rule_id"]
        result["check"] = f"{group}.{check}"
        result["policy_sections"] = definition["sources"]
        result.update(details)
        return result

    # Apply shared human-review checks first, then evaluate the selected operation.
    def check(self, operation: str, amount: Any = None, facts: dict | None = None) -> dict:
        if operation not in OPERATIONS:
            return dict(self.unsupported_operation)
        if facts is not None and not isinstance(facts, dict):
            raise ValueError("facts must be a dictionary of recorded values")
        facts = facts or {}
        review = self._human_review(facts)
        if review is not None:
            return review
        if operation == "refund":
            return self._refund(amount, facts)
        return getattr(self, f"_{operation}")(facts)

    # -------------------- Group 1: Shared human review --------------------

    # Return the matching JSON outcome; no trigger means continue to the operation.
    def _human_review(self, facts: dict) -> dict | None:
        group, config = "human_review", self.data["human_review"]
        if any(name in facts and type(facts[name]) is not bool for name in ("disputed", "unsafe_item", "human_requested")):
            return self._result(group, "invalid_risk_flags")
        if facts.get("disputed") is True:
            return self._result(group, "disputed_facts")
        if facts.get("unsafe_item") is True:
            return self._result(group, "unsafe_goods")
        if facts.get("human_requested") is True:
            return self._result(group, "human_requested")
        if facts.get("evidence_status") in config["evidence_statuses"]:
            return self._result(group, "evidence_unavailable")
        if facts.get("reason") in config["return_reasons"]:
            return self._result(group, "counterfeit_claim")
        if facts.get("item_type") in config["item_types"]:
            return self._result(group, "unsupported_item_type")
        return None

    # -------------------- Group 2: Refund --------------------

    # Check currency, approval limit, payment prerequisites and available balance.
    def _refund(self, amount: Any, facts: dict) -> dict:
        group, config = "refund", self.data["refund"]
        value = _money(amount)
        if value <= 0:
            raise ValueError("refund amount must be positive")
        details = {"limit": float(self.refund_limit), "currency": self.currency}
        if facts.get("currency") is not None and facts["currency"] != self.currency:
            return self._result(group, "currency_mismatch", **details)
        if value > self.refund_limit:
            return self._result(group, "limit_exceeded", **details)
        missing = sorted(key for key in config["required_facts"] if facts.get(key) is None)
        if missing:
            return self._result(group, "missing_facts", missing_facts=missing, **details)
        if facts["payment_status"] != config["required_payment_status"]:
            return self._result(group, "payment_not_captured", **details)
        if facts["original_payment_method"] is not True:
            return self._result(group, "different_payment_method", **details)
        if facts["request_accepted"] is not True:
            return self._result(group, "request_not_accepted", **details)
        if facts["inspection_or_waiver"] is not True:
            return self._result(group, "inspection_incomplete", **details)
        balance = _money(facts["captured_amount"]) - _money(facts["refunded_amount"]) - _money(facts["pending_refund_amount"])
        if value > balance:
            return self._result(group, "balance_exceeded", **details)
        if value > _money(facts["approved_refund_amount"]):
            return self._result(group, "approved_amount_exceeded", **details)
        return self._result(group, "passed", **details)

    # -------------------- Group 3: Return request --------------------

    # Check the affected item's dates, reason, evidence and change-of-mind conditions.
    def _return_request(self, facts: dict) -> dict:
        group, config = "return_request", self.data["return_request"]
        if not facts.get("delivered_at") or not facts.get("requested_at"):
            return self._result(group, "missing_dates")
        delivered = _date(facts["delivered_at"], self.timezone)
        requested = _date(facts["requested_at"], self.timezone)
        deadline = delivered + timedelta(days=config["window_calendar_days"])
        if requested < delivered:
            return self._result(group, "request_before_delivery")
        if requested > deadline or (requested == deadline and not config["deadline_inclusive"]):
            return self._result(group, "window_exceeded", deadline=deadline.isoformat())
        if facts.get("reason") not in config["allowed_reasons"]:
            return self._result(group, "unsupported_reason")
        if facts.get("evidence_status") != "verified":
            return self._result(group, "unverified_evidence")
        if facts["reason"] == "change_of_mind":
            names = ["final_sale", *config["change_of_mind_required"]]
            if any(type(facts.get(name)) is not bool for name in names):
                return self._result(group, "missing_item_conditions")
            if facts["final_sale"]:
                return self._result(group, "final_sale")
            if not all(facts[name] for name in config["change_of_mind_required"]):
                return self._result(group, "item_conditions_failed")
        return self._result(group, "passed", deadline=deadline.isoformat())

    # -------------------- Group 4: Return dispatch --------------------

    # Check authorization, actual carrier handover and a recorded approved extension.
    def _return_dispatch(self, facts: dict) -> dict:
        group, config = "return_dispatch", self.data["return_dispatch"]
        if not facts.get("approved_at") or not facts.get("handed_over_at"):
            return self._result(group, "missing_dates")
        approved = _date(facts["approved_at"], self.timezone)
        sent = _date(facts["handed_over_at"], self.timezone)
        deadline = approved + timedelta(days=config["window_calendar_days"])
        if facts.get("extension_until"):
            if facts.get("extension_approved") is not True:
                return self._result(group, "unapproved_extension")
            deadline = _date(facts["extension_until"], self.timezone)
        if facts.get("carrier_accepted") is not True:
            return self._result(group, "handover_unverified")
        if sent < approved:
            return self._result(group, "dispatch_before_approval")
        if sent > deadline or (sent == deadline and not config["deadline_inclusive"]):
            return self._result(group, "deadline_exceeded", deadline=deadline.isoformat())
        return self._result(group, "passed", deadline=deadline.isoformat())

    # -------------------- Group 5: Cancellation --------------------

    # Check status, then require every fulfillment blocker to be verified and false.
    def _cancel(self, facts: dict) -> dict:
        group, config = "cancel", self.data["cancel"]
        if facts.get("order_status") == "cancelled":
            return self._result(group, "already_cancelled")
        if facts.get("order_status") != config["allowed_order_status"]:
            return self._result(group, "wrong_order_status")
        if any(type(facts.get(name)) is not bool for name in config["blockers"]):
            return self._result(group, "unverified_fulfillment")
        if any(facts[name] for name in config["blockers"]):
            return self._result(group, "fulfillment_started")
        return self._result(group, "passed")

    # -------------------- Group 6: Delivery --------------------

    # Handle immediate-review situations before calculating the ordinary waiting period.
    def _delivery(self, facts: dict) -> dict:
        group, config = "delivery", self.data["delivery"]
        status = facts.get("delivery_status")
        if status in config["immediate_human_statuses"]:
            return self._result(group, "immediate_investigation")
        if status == "delivered":
            return self._result(group, "delivered")
        if status not in config["ordinary_statuses"]:
            return self._result(group, "unknown_status")
        if not facts.get("estimated_delivery") or not facts.get("as_of"):
            return self._result(group, "missing_dates")
        days_late = (_date(facts["as_of"], self.timezone) - _date(facts["estimated_delivery"], self.timezone)).days
        if days_late > config["investigate_after_calendar_days"]:
            return self._result(group, "waiting_period_exceeded", days_late=days_late)
        return self._result(group, "passed", days_late=days_late)
