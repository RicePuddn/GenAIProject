"""The one rules engine. Python evaluates the checks; the LLM receives the result.

`Rules` runs the shared human-review checks, then only the requested operation; every decision copies
its complete outcome from rules.json. `BackendFacts` collects the recorded facts those checks run on,
read from the database and never taken from the model. Every setup's writes are checked here, through
the write gate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from SupportOpsBot.runtime.shared.clock import Clock
from SupportOpsBot.runtime.shared.config import ROOT
from SupportOpsBot.runtime.shared.db import PostgresReader

DEFAULT_RULES_FILE = ROOT / "data/policy/rules.json"
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
    if outcome["requires_human"]:
        expected_routes = {"human"}
    elif outcome["allowed"]:
        expected_routes = {"auto", "approver"}
    else:
        expected_routes = {None}
    if outcome.get("route") not in expected_routes:
        raise ValueError("rule route must agree with allowed and requires_human")
    if not isinstance(outcome.get("reason"), str) or not outcome["reason"].strip():
        raise ValueError("rule outcomes need a reason")


@dataclass(frozen=True)
class Rules:
    version: str
    currency: str
    refund_limit: Decimal
    refund_auto_limit: Decimal
    timezone: str
    data: dict[str, Any]
    unsupported_operation: dict

    # -------------------- Load configuration and build results --------------------

    # Validate the JSON checks and their reviewed policy-PDF fingerprints.
    @classmethod
    def load(cls, path: str | Path = DEFAULT_RULES_FILE) -> Rules:
        from SupportOpsBot.runtime.shared.policy.rag.identifiers import policy_source

        path = Path(path)
        try:
            config = json.loads((path if path.is_absolute() else ROOT / path).read_text())
            data = config["groups"]
            if set(data) != set(GROUPS):
                raise ValueError("rules.json must contain the six supported rule groups")
            limit = _money(data["refund"]["max_reviewed_amount"])
            auto_limit = _money(data["refund"]["max_auto_amount"])
            if not 0 < auto_limit <= limit:
                raise ValueError("refund limits must satisfy 0 < max_auto_amount <= max_reviewed_amount")
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
                source = ROOT / "data/policy/pdf" / name
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"{name} changed: review its hard rules before updating the fingerprint")
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"invalid rules file {path}: {exc!r}") from exc
        return cls(
            version=str(config["version"]),
            currency=config["currency"],
            refund_limit=limit,
            refund_auto_limit=auto_limit,
            timezone=config["timezone"],
            data=data,
            unsupported_operation=config["unsupported_operation"],
        )

    @property
    def return_reasons(self) -> list[str]:
        return self.data["return_request"]["allowed_reasons"]

    # Copy the JSON outcome, then add the rule identity, policy sources and case details.
    def _result(self, group: str, check: str, **details) -> dict:
        definition = self.data[group]
        outcome = definition["checks"][check]["outcome"]

        result = dict(outcome)
        result["rule"] = definition["rule_id"]
        result["check"] = f"{group}.{check}"
        # A rule's sources are not retrieved policy text, so they cannot be cited for a write.
        result["rule_sources"] = definition["sources"]
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
        if facts.get("disputed") is True:
            return self._result(group, "disputed_facts")
        if facts.get("evidence_status") in config["evidence_statuses"]:
            return self._result(group, "evidence_unavailable")
        if facts.get("reason") in config["return_reasons"]:
            return self._result(group, "counterfeit_claim")
        if facts.get("item_type") in config["item_types"]:
            return self._result(group, "unsupported_item_type")
        return None

    # -------------------- Group 2: Refund --------------------

    # Step 1: Check refund prerequisites. Step 2: Select the JSON outcome for its amount.
    def _refund(self, amount: Any, facts: dict) -> dict:
        group, config = "refund", self.data["refund"]
        value = _money(amount)
        if value <= 0:
            raise ValueError("refund amount must be positive")
        details = {"limit": float(self.refund_limit), "currency": self.currency,
                   "return_id": facts.get("return_id"), "payment_id": facts.get("payment_id")}
        if facts.get("currency") is not None and facts["currency"] != self.currency:
            return self._result(group, "currency_mismatch", **details)
        missing = sorted(key for key in config["required_facts"] if facts.get(key) is None)
        if missing:
            return self._result(group, "missing_facts", missing_facts=missing, **details)
        if facts["payment_status"] != config["required_payment_status"]:
            return self._result(group, "payment_not_captured", **details)
        if facts["request_accepted"] is not True:
            return self._result(group, "request_not_accepted", **details)
        if facts["inspection_or_waiver"] is not True:
            return self._result(group, "inspection_incomplete", **details)
        balance = _money(facts["captured_amount"]) - _money(facts["refunded_amount"]) - _money(facts["pending_refund_amount"])
        if value > balance:
            return self._result(group, "balance_exceeded", **details)
        if value > _money(facts["approved_refund_amount"]):
            return self._result(group, "approved_amount_exceeded", **details)
        if facts.get("goods_value") is not None and value > _money(facts["goods_value"]):
            return self._result(group, "goods_value_exceeded", **details)
        if value > self.refund_limit:
            return self._result(group, "limit_exceeded", **details)
        if value > self.refund_auto_limit:
            return self._result(group, "automated_review", **details)
        return self._result(group, "auto_approval", **details)

    # -------------------- Group 3: Return request --------------------

    # Check the affected item's dates, reason, evidence and change-of-mind conditions.
    def _return_request(self, facts: dict) -> dict:
        group, config = "return_request", self.data["return_request"]
        if facts.get("return_scope_unverified") is True:
            return self._result(group, "return_scope_unverified")
        if facts.get("existing_return") is True:
            return self._result(group, "existing_return")
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
        if facts.get("new_request") is True:
            return self._new_return_request(facts, deadline)
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

    # A first request has no return record yet, so its evidence and item conditions are the store's to
    # review later; only what is already recorded can refuse it here.
    def _new_return_request(self, facts: dict, deadline: date) -> dict:
        group = "return_request"
        if facts["reason"] == "change_of_mind":
            if type(facts.get("final_sale")) is not bool:
                return self._result(group, "missing_item_conditions")
            if facts["final_sale"]:
                return self._result(group, "final_sale")
        return self._result(group, "new_request", deadline=deadline.isoformat())

    # -------------------- Group 4: Return dispatch --------------------

    # Check authorization, actual carrier handover and a recorded approved extension.
    def _return_dispatch(self, facts: dict) -> dict:
        group, config = "return_dispatch", self.data["return_dispatch"]
        if not facts.get("authorized_at") or not facts.get("handed_over_at"):
            return self._result(group, "missing_dates")
        authorized = _date(facts["authorized_at"], self.timezone)
        sent = _date(facts["handed_over_at"], self.timezone)
        deadline = authorized + timedelta(days=config["window_calendar_days"])
        if facts.get("extension_until"):
            if facts.get("extension_approved") is not True:
                return self._result(group, "unapproved_extension")
            deadline = _date(facts["extension_until"], self.timezone)
        if facts.get("carrier_accepted") is not True:
            return self._result(group, "handover_unverified")
        if sent < authorized:
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
        if status in {"delivered", "cancelled"}:
            return self._result(group, status)
        if status not in config["ordinary_statuses"]:
            return self._result(group, "unknown_status")
        if not facts.get("estimated_delivery") or not facts.get("as_of"):
            return self._result(group, "missing_dates")
        days_late = (_date(facts["as_of"], self.timezone) - _date(facts["estimated_delivery"], self.timezone)).days
        if days_late > config["investigate_after_calendar_days"]:
            return self._result(group, "waiting_period_exceeded", days_late=days_late)
        return self._result(group, "passed", days_late=days_late)


# -------------------- Recorded facts for the checks --------------------

VOID_REFUND_STATUSES = {"failed", "cancelled", "rejected"}
VOID_RETURN_STATUSES = {"cancelled", "rejected", "closed"}
# Returns the store still has to act on.
AWAITING_STORE_RETURN_STATUSES = {"requested", "received"}
# Rule-engine operation names and the corresponding database write names.
WRITE_OPERATIONS = {"refund": "refund", "return_request": "return", "cancel": "order_change"}


# The paid line amount includes its recorded discount and tax allocation.
def goods_value(item: dict, quantity: int | None = None) -> Decimal:
    paid = item.get("paid_amount")
    total = Decimal(str(paid)) if paid is not None else Decimal(str(item["unit_price"])) * item["quantity"]
    selected_quantity = item["quantity"] if quantity is None else quantity
    return (total * selected_quantity / item["quantity"]).quantize(Decimal("0.01"))


class BackendFacts:
    def __init__(self, reader: PostgresReader, clock: Clock):
        self.reader, self.clock = reader, clock

    # Read the affected goods, then collect only the fields needed by the operation.
    def read(self, operation: str, order_id: str, item_ids: list[str] | None = None,
             *, history: dict | None = None, stated_reason: str | None = None) -> dict[str, Any]:
        order = self.reader.read("orders", order_id)
        if not isinstance(order, dict) or order.get("not_found"):
            return {}
        if operation not in {"cancel", "refund", "return_request", "return_dispatch", "exchange"}:
            return {}
        items = self.reader.items_for_order(order_id)
        selected = set(item_ids or [item["id"] for item in items])
        if not selected or not selected <= {item["id"] for item in items}:
            return {"order_status": order.get("status")} if operation == "cancel" else {}
        affected = [item for item in items if item["id"] in selected]
        shipments = self.reader.shipments_for_order(order_id)
        if operation == "cancel":
            return self._cancellation(order, items, shipments)

        # Resolve delivery using parcel-to-item links. Ambiguous historical split parcels stay unknown.
        parcels = [row for row in shipments if selected.intersection(
            line["order_item_id"] for line in row.get("items", []))]
        if not parcels and len(shipments) == 1:
            parcels = shipments
        covered = {line["order_item_id"] for row in parcels for line in row.get("items", [])}
        facts: dict[str, Any] = {}
        if parcels and all(row.get("delivered_at") for row in parcels) and (selected <= covered or len(shipments) == 1):
            facts["delivered_at"] = min(row["delivered_at"] for row in parcels)
        types = {item.get("product_type") for item in affected}
        if None not in types:
            facts["item_type"] = next((kind for kind in types if kind != "physical"), "physical")
        history = history if history is not None else self.reader.returns_for_order(order_id)
        facts.update(self._return_review(history, selected, affected, operation))
        if operation in {"return_request", "exchange"}:
            facts.setdefault("requested_at", self.clock.now().isoformat())
        if operation == "return_request" and stated_reason and "return_id" not in facts \
                and not facts["return_scope_unverified"]:
            # The first request for these goods: until a return exists, the reason can only be the one stated.
            facts.update(new_request=True, reason=stated_reason)
            if all(type(item.get("final_sale")) is bool for item in affected):
                facts["final_sale"] = any(item["final_sale"] for item in affected)
        if operation == "refund":
            if not facts.get("delivered_at") and facts.get("inspection_status") != "waived":
                facts["inspection_or_waiver"] = False
            whole_order = selected == {item["id"] for item in items}
            facts.update(self._payment(order, affected, history.get("refunds", []), whole_order))
            if order.get("status") == "cancelled" and whole_order and "return_id" not in facts:
                # ODR-OrderChangeCancel-4.2: a cancelled order owes its captured balance, with no return.
                facts.update(request_accepted=True, inspection_or_waiver=True, approved_refund_amount=(
                    facts["captured_amount"] - facts["refunded_amount"] - facts["pending_refund_amount"]))
        return facts

    # Order-item states describe packing; shipment records describe label creation and handover.
    def _cancellation(self, order: dict, items: list[dict], shipments: list[dict]) -> dict:
        facts = {"order_status": order.get("status")}
        states = {item.get("item_status") for item in items}
        if states <= {"open", "packing", "packed", "fulfilled", "cancelled"}:
            facts["packing_started"] = bool(states.intersection({"packing", "packed", "fulfilled"}))
            facts["any_quantity_fulfilled"] = "fulfilled" in states
        known = {"processing", "label_created", "shipped", "in_transit", "delayed", "delivery_attempted",
                 "out_for_delivery", "delivered", "cancelled"}
        if all(row.get("status") in known for row in shipments):
            facts["label_created"] = any(row.get("label_created_at") or row.get("status") == "label_created"
                                         for row in shipments)
            facts["carrier_collected"] = any(row.get("shipped_at") or row.get("delivered_at") or row.get("status")
                in {"shipped", "in_transit", "delayed", "delivery_attempted", "out_for_delivery", "delivered"}
                for row in shipments)
        return facts

    # Use one matching return review; never apply a different item's findings to the selected goods.
    def _return_review(self, history: dict, selected: set[str], items: list[dict], operation: str) -> dict:
        active = [row for row in history.get("returns", []) if row.get("status") not in VOID_RETURN_STATUSES]
        matches = [row for row in active if selected <= {line["order_item_id"] for line in row.get("items", [])}]
        if len(matches) != 1:
            return {"return_scope_unverified": bool(active)}
        request = matches[0]
        lines = [line for line in request["items"] if line["order_item_id"] in selected]
        accepted = request.get("status") == "approved" and bool(request.get("approved_at"))
        facts = {
            "return_id": request["id"], "reason": request.get("reason"),
            "requested_at": request.get("requested_at"), "request_accepted": accepted,
            "disputed": request.get("dispute_status") == "open",
            "alternative_resolution_agreed": accepted and request.get("resolution") == "exchange",
        }
        evidence = request.get("evidence_status")
        if evidence is not None:
            if evidence == "verified" and not request.get("reviewed_at"):
                evidence = "pending"
            facts["evidence_status"] = evidence
        if request.get("inspection_status") is not None:
            facts["inspection_status"] = request["inspection_status"]
            facts["inspection_or_waiver"] = bool(request.get("reviewed_at")) and request["inspection_status"] in {"passed", "waived"}
        # Completed and pending payments consume this return's approval as well as the order balance.
        refunds = [row for row in history.get("refunds", []) if row.get("return_id") == request["id"]
                   and row.get("status") not in VOID_REFUND_STATUSES] if operation == "refund" else []
        full_return = selected == {line["order_item_id"] for line in request["items"]}
        # An earlier refund has no per-item allocation; leave a partial return's remaining amount unknown.
        if (not refunds or full_return) and all(line.get("approved_refund_amount") is not None for line in lines):
            approved = sum((Decimal(str(line["approved_refund_amount"])) for line in lines), Decimal(0))
            used = sum((Decimal(str(row["amount"])) for row in refunds), Decimal(0))
            facts["approved_refund_amount"] = max(Decimal(0), approved - used)
        for name in ("unused", "original_packaging", "tags_complete", "accessories_complete"):
            if all(type(line.get(name)) is bool for line in lines):
                facts[name] = all(line[name] for line in lines)
        if all(type(item.get("final_sale")) is bool for item in items):
            facts["final_sale"] = any(item["final_sale"] for item in items)
        if operation == "return_request":
            facts["existing_return"] = request.get("status") in {"approved", "in_transit", "received"}
        for name in ("authorized_at", "handed_over_at", "extension_until"):
            if request.get(name):
                facts[name] = request[name]
        facts["extension_approved"] = bool(request.get("extension_approved_at"))
        facts["carrier_accepted"] = bool(request.get("handed_over_at"))
        return facts

    # Derive balances from original captured payments; callers cannot select a different destination.
    def _payment(self, order: dict, items: list[dict], refunds: list[dict], whole_order: bool) -> dict:
        payments = self.reader.payments_for_order(order["id"])
        captured = [row for row in payments if row.get("status") == "captured"]
        value = sum((goods_value(item) for item in items), Decimal(0))
        if order.get("status") == "cancelled" and whole_order:
            value = Decimal(str(order["total"]))
        return {
            "currency": order.get("currency"), "payment_status": "captured" if captured else "not_captured",
            "payment_id": captured[0]["id"] if len(captured) == 1 else None,
            "captured_amount": sum((Decimal(str(row["amount"])) for row in captured), Decimal(0)),
            "refunded_amount": sum((Decimal(str(row["amount"])) for row in refunds if row.get("status") == "processed"), Decimal(0)),
            "pending_refund_amount": sum((Decimal(str(row["amount"])) for row in refunds
                 if row.get("status") not in VOID_REFUND_STATUSES | {"processed"}), Decimal(0)),
            "goods_value": value,
        }

    # Check every parcel independently; one delivered parcel does not settle another's delay.
    def deliveries(self, order_id: str) -> list[dict[str, Any]]:
        return [{"delivery_status": row.get("status"), "estimated_delivery": row.get("estimated_delivery"),
                 "as_of": self.clock.now().isoformat()} for row in self.reader.shipments_for_order(order_id)]
