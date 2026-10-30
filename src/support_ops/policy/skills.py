"""Progressive disclosure of policy skills."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import model_validator

from support_ops.helpers.arguments import NonEmpty, ToolArguments
from support_ops.helpers.config import ROOT
from support_ops.policy.provider import ScopedPolicyTool
from support_ops.policy.rag.loader import MARKDOWN_CLAUSE
from support_ops.policy.rules import Rules
from support_ops.systems.tools import PolicyArguments


class PolicySkillsArguments(ToolArguments):
    """list: your policy skills; open: read a skill's SKILL.md; read: read one file of that skill. scope own
    (the default) keeps to your skills; use all when the case crosses into another area."""
    action: Literal["list", "open", "read"]
    skill: NonEmpty | None = None
    file: NonEmpty | None = None
    scope: Literal["own", "all"] = "own"

    @model_validator(mode="after")
    def required_for_action(self) -> Self:
        if self.action in {"open", "read"} and not self.skill:
            raise ValueError(f"{self.action} requires skill")
        if self.action == "read" and not self.file:
            raise ValueError("read requires file")
        return self


class SkillsPolicy:
    arguments = PolicyArguments
    description = PolicyArguments.__doc__
    prompt_hint = ("Discover policies, open the matching SKILL.md, then read the relevant clause "
                   "before advising. Never invent policy filenames or missing facts.")
    scope_hint = "Your policy skills cover your own area; use scope all when the case crosses into another area."
    scope_setting = "policy_skills"
    strict_scope = True   # a configured skill folder that does not exist is a setup error

    def __init__(self, root: str | Path = ROOT / "policy/skills", rules: Rules | None = None):
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

    def has_skill(self, name: str) -> bool:
        """Whether `name` is a skill folder with a SKILL.md, without reading every skill."""
        base = (self.root / name).resolve()
        return base.parent == self.root and not (self.root / name).is_symlink() and (base / "SKILL.md").is_file()

    def check_scope(self, scope: Sequence[str] | None) -> list[str]:
        return [skill for skill in scope or [] if not self.has_skill(skill)]

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

    @staticmethod
    def citable(result: dict[str, Any]) -> set[str]:
        """A file read can be cited by its name, skill/name, or a clause ID heading in it. Every skill has a
        SKILL.md, so that one needs its skill: a bare SKILL.md does not say which policy was read."""
        if "file" not in result or "content" not in result:
            return set()
        ids = {match.group(1) for match in MARKDOWN_CLAUSE.finditer(result["content"])}
        names = {f"{result['skill']}/{result['file']}"} | ({result["file"]} if result["file"] != "SKILL.md" else set())
        return names | ids

    def scoped_tool(self, scope: Sequence[str] | None) -> ScopedPolicyTool:
        """list/open/read without `check`, limited by default to the specialist's `scope` skill folders."""
        def own(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [item for item in skills if not scope or item["skill"] in scope]

        def run(ticket_id: str, args: dict[str, Any]) -> dict[str, Any]:
            everything = args.get("scope", "own") == "all" or not scope
            if args["action"] == "list":
                catalog = self.catalog() if everything else own(self.catalog())
                result = {"skills": [item["skill"] for item in catalog], "catalog": catalog,
                          "scope": "all" if everything else "own"}
                if not catalog:
                    result["note"] = "No policy skills are available here; hand the case over rather than guess."
                return result
            skill = args["skill"]
            if not self.has_skill(skill):
                # A guessed name: offer only names that exist, never another area's.
                choices = ([item["skill"] for item in self.catalog()] if everything
                           else [name for name in scope if self.has_skill(name)])
                raise ValueError(f"no policy skill named {skill!r}; "
                                 + (f"your skills: {choices}" if choices else "no policy skills are available"))
            if not everything and skill not in scope:
                raise ValueError(f"{skill} is outside your policy skills {list(scope)}; "
                                 "use scope all if the case crosses into another area")
            return self.run({"action": args["action"], "skill": skill,
                             **({"file": args["file"]} if args.get("file") else {})})

        def context() -> str:
            # Only the specialist's own skills, so a plan does not list reads the default scope refuses.
            return ("Your policy skills (contents must still be read with tools; scope all reaches the "
                    "others): " + json.dumps(own(self.catalog())))

        return ScopedPolicyTool(PolicySkillsArguments, PolicySkillsArguments.__doc__, run,
                                self.prompt_hint, self.scope_hint, context, self.citable)
