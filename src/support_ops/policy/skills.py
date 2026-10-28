"""Progressive disclosure of policy skills and a shared mock approval rule."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from support_ops.helpers.config import ROOT

REFUND_LIMIT = Decimal("100.00")


def check_operation(operation: str, amount: Any) -> dict[str, Any]:
    """This prototype only approves simulated refunds in SGD."""
    if operation != "refund":
        return {"allowed": False, "requires_human": True, "reason": "unsupported mock operation"}
    if amount is None:
        raise ValueError("refund amount is required")
    value = Decimal(str(amount))
    if not value.is_finite() or value <= 0:
        raise ValueError("refund amount must be finite and positive")
    return {
        "allowed": value <= REFUND_LIMIT,
        "requires_human": value > REFUND_LIMIT,
        "rule": "mock_refund_limit_100", "limit": float(REFUND_LIMIT), "currency": "SGD",
    }


class SkillsPolicy:
    def __init__(self, root: str | Path = ROOT / "policy/skills"):
        path = Path(root)
        self.root = (path if path.is_absolute() else ROOT / path).resolve()

    def catalog(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        catalog = []
        for path in sorted(self.root.iterdir()):
            entrypoint = path / "SKILL.md"
            if not path.is_dir() or path.is_symlink() or not entrypoint.is_file():
                continue
            content = entrypoint.read_text()
            metadata = {}
            if content.startswith("---\n"):
                parts = content.split("---", 2)
                if len(parts) == 3:
                    metadata = yaml.safe_load(parts[1]) or {}
            catalog.append({
                "skill": path.name, "description": str(metadata.get("description", "")),
                "entrypoint": "SKILL.md",
                "files": sorted(file.name for file in path.glob("*.md") if file.is_file()),
            })
        return catalog

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        action = args.get("action")
        if action == "check":
            return check_operation(args.get("operation", ""), args.get("amount"))
        if action == "list":
            catalog = self.catalog()
            return {"skills": [item["skill"] for item in catalog], "catalog": catalog}
        if action not in {"open", "read"}:
            raise ValueError("unsupported policy action")

        skill = args.get("skill") or args.get("file", "")
        filename = "SKILL.md" if action == "open" or not args.get("skill") else args.get("file", "SKILL.md")
        # Accept common entrypoint aliases while exposing the canonical filename.
        if filename.lower() in {"policy.md", "skill.md", "index.md"} or filename == skill:
            filename = "SKILL.md"
        base = (self.root / skill).resolve()
        path = (base / filename).resolve()
        if base.parent != self.root or base == self.root or not path.is_relative_to(base):
            raise ValueError("invalid policy path")
        if path.suffix != ".md" or not path.is_file():
            available = sorted(file.name for file in base.glob("*.md") if file.is_file())
            raise FileNotFoundError(f"policy file not found: {skill}/{filename}; available files: {available}")
        return {"skill": skill, "file": str(path.relative_to(base)), "content": path.read_text()}
