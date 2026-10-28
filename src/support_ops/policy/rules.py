"""Shared hard policy limits from policy/rules.yaml"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from support_ops.helpers.config import ROOT

DEFAULT_RULES_FILE = ROOT / "policy/rules.yaml"


@dataclass(frozen=True)
class Rules:
    version: str
    currency: str
    refund_rule_id: str
    refund_limit: Decimal

    @classmethod
    def load(cls, path: str | Path = DEFAULT_RULES_FILE) -> Rules:
        path = Path(path)
        data = yaml.safe_load((path if path.is_absolute() else ROOT / path).read_text()) or {}
        try:
            refund = data["refund"]
            limit = Decimal(str(refund["max_auto_amount"]))
            rules = cls(str(data["version"]), str(data["currency"]), str(refund["rule_id"]), limit)
        except (KeyError, TypeError, InvalidOperation) as exc:
            raise ValueError(f"invalid rules file {path}: {exc!r}") from exc
        if not limit.is_finite() or limit <= 0:
            raise ValueError(f"invalid rules file {path}: refund limit must be positive")
        return rules

    def check(self, operation: str, amount: Any) -> dict[str, Any]:
        """This prototype only approves simulated refunds."""
        if operation != "refund":
            return {"allowed": False, "requires_human": True, "reason": "unsupported mock operation"}
        if amount is None:
            raise ValueError("refund amount is required")
        value = Decimal(str(amount))
        if not value.is_finite() or value <= 0:
            raise ValueError("refund amount must be finite and positive")
        return {
            "allowed": value <= self.refund_limit,
            "requires_human": value > self.refund_limit,
            "rule": self.refund_rule_id, "limit": float(self.refund_limit), "currency": self.currency,
        }
