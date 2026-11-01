"""Progressive disclosure of policy skills."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, model_validator

from support_ops.shared.config import ROOT
from support_ops.shared.rules import Rules
from support_ops.shared.tools import CHECK_DESCRIPTION, NonEmpty, ToolArguments

# Clause headings belong to skills Markdown; the RAG loader reads numbered PDF sections.
MARKDOWN_CLAUSE = re.compile(r"^#{1,6}\s+([A-Z]+(?:-[A-Za-z0-9]+)+(?:\.[0-9]+)*)\s+\S.*$", re.MULTILINE)


class PolicyArguments(ToolArguments):
    __doc__ = "list: discover skills; open: read SKILL.md; read: read a discovered clause; " + CHECK_DESCRIPTION
    action: Literal["list", "open", "read", "check"]
    skill: NonEmpty | None = None
    file: NonEmpty | None = None
    operation: Literal["delivery"] | None = None
    order_id: NonEmpty | None = Field(default=None, description="The order whose parcels a delivery check covers.")

    @model_validator(mode="after")
    def required_for_action(self) -> Self:
        if self.action == "open" and not self.skill:
            raise ValueError("open requires skill")
        if self.action == "read" and not (self.skill or self.file):
            raise ValueError("read requires skill or file")
        if self.action == "check" and not (self.operation and self.order_id):
            raise ValueError("check requires operation delivery and order_id")
        return self


class SkillsPolicy:
    arguments = PolicyArguments
    description = PolicyArguments.__doc__
    prompt_hint = ("Discover policies, open the matching SKILL.md, then read the relevant clause "
                   "before advising. Never invent policy filenames or missing facts.")

    def __init__(self, root: str | Path = ROOT / "data/policy/skills", rules: Rules | None = None):
        path = Path(root)
        self.root = (path if path.is_absolute() else ROOT / path).resolve()
        self.rules = rules or Rules.load()

    def warm_up(self) -> None:
        """Skills read files on demand, so this only checks they can be read: a missing folder or an
        unreadable SKILL.md fails at start-up instead of during a ticket."""
        if not self.root.is_dir():
            raise ValueError(f"policy skills folder not found: {self.root}")
        catalog = self.catalog()
        if not catalog:
            raise ValueError(f"no policy skills (folders with a SKILL.md) in {self.root}")
        if broken := [item["skill"] for item in catalog if item.get("error")]:
            raise ValueError(f"unreadable SKILL.md front matter in {broken}")

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
            metadata, error = {}, None
            if content.startswith("---\n"):
                parts = content.split("---", 2)
                if len(parts) == 3:
                    try:
                        metadata = yaml.safe_load(parts[1]) or {}
                    except yaml.YAMLError:
                        # One bad file must not take every skill down; warm_up reports it at start-up.
                        error = "SKILL.md front matter could not be read"
                    if not isinstance(metadata, dict):
                        metadata, error = {}, "SKILL.md front matter is not a mapping"
            catalog.append({
                "skill": path.name, "description": str(metadata.get("description", "")),
                "entrypoint": "SKILL.md",
                "files": sorted(file.name for file in path.glob("*.md") if file.is_file()),
                **({"error": error} if error else {}),
            })
        return catalog

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        action = args.get("action")
        if action == "check":
            return self.rules.check(args.get("operation", ""), args.get("amount"), args.get("facts"))
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

    @staticmethod
    def citable(result: dict[str, Any]) -> set[str]:
        """A file read can be cited by its name, skill/name, or a clause ID heading in it. Every skill has a
        SKILL.md, so that one needs its skill: a bare SKILL.md does not say which policy was read."""
        if "file" not in result or "content" not in result:
            return set()
        ids = {match.group(1) for match in MARKDOWN_CLAUSE.finditer(result["content"])}
        names = {f"{result['skill']}/{result['file']}"} | ({result["file"]} if result["file"] != "SKILL.md" else set())
        return names | ids
