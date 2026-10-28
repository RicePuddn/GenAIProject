"""Progressive disclosure of policy skills."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from support_ops.helpers.config import ROOT
from support_ops.policy.rules import Rules
from support_ops.systems.tools import PolicyArguments


class SkillsPolicy:
    arguments = PolicyArguments
    description = PolicyArguments.__doc__
    prompt_hint = ("Discover policies, open the matching SKILL.md, then read the relevant clause "
                   "before advising. Never invent policy filenames or missing facts.")

    def __init__(self, root: str | Path = ROOT / "policy/skills", rules: Rules | None = None):
        path = Path(root)
        self.root = (path if path.is_absolute() else ROOT / path).resolve()
        self.rules = rules or Rules.load()

    def warm_up(self) -> None:
        """Skills read files on demand; there is nothing to load up front."""

    def planner_context(self) -> str:
        return ("Available policy filenames (contents must still be read with tools): "
                + json.dumps(self.catalog()))

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
            return self.rules.check(args.get("operation", ""), args.get("amount"))
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
