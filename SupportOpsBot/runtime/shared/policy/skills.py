"""Progressive disclosure of policy skills."""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, model_validator

from SupportOpsBot.runtime.shared.config import ROOT
from SupportOpsBot.runtime.shared.rules import Rules
from SupportOpsBot.services.tools.read.policy import CHECK_DESCRIPTION
from SupportOpsBot.services.structure.structure import NonEmpty, ToolArguments

# Clause headings belong to skills Markdown; the RAG loader reads numbered PDF sections.
MARKDOWN_CLAUSE = re.compile(r"^#{1,6}\s+([A-Z]+(?:-[A-Za-z0-9]+)+(?:\.[0-9]+)*)\s+\S.*$", re.MULTILINE)


class PolicyArguments(ToolArguments):
    __doc__ = ("list: discover skills (pass only action); open: read the skill's SKILL.md (pass action and skill, "
               "never file); read: read a clause file listed in SKILL.md (pass action, skill and exact file; "
               "never SKILL.md); " + CHECK_DESCRIPTION)
    action: Literal["list", "open", "read", "check"] = Field(description=(
        "list takes only action; open takes action and skill, with no file; "
        "read takes action, skill and a clause filename listed in SKILL.md; check takes operation and order_id."))
    skill: NonEmpty | None = Field(default=None, description="Required for open and read; omit for list and check.")
    file: NonEmpty | None = Field(default=None, description=(
        "For read only: the exact clause filename listed in SKILL.md (such as eligibility.md); "
        "omit for open, which always reads SKILL.md."))
    operation: Literal["delivery"] | None = None
    order_id: NonEmpty | None = Field(default=None, description="The order whose parcels a delivery check covers.")

    @model_validator(mode="after")
    def required_for_action(self) -> Self:
        if self.action == "list" and (self.skill or self.file):
            raise ValueError("list accepts only action; omit skill and file")
        if self.action == "open" and not self.skill:
            raise ValueError("open requires skill; retry with action and skill only")
        if self.action == "open" and self.file:
            raise ValueError("retry open with action and skill only; open reads SKILL.md. Use read only for a clause filename listed there")
        if self.action == "read" and not self.skill:
            raise ValueError("read requires skill and file; use the skill name from list")
        if self.action == "read" and not self.file:
            raise ValueError("read requires both skill and file; use the filename from SKILL.md")
        if self.action == "read" and self.file and (
            self.file.casefold() in {"skill.md", "policy.md", "index.md"} or self.file == self.skill
        ):
            raise ValueError("read requires a clause filename listed in SKILL.md; use open with action and skill only to read SKILL.md")
        if self.action == "check":
            if self.skill or self.file:
                raise ValueError("check accepts operation and order_id; omit skill and file")
            if not (self.operation and self.order_id):
                raise ValueError("check requires operation delivery and order_id")
        elif self.operation or self.order_id:
            raise ValueError(f"{self.action} does not accept operation or order_id")
        return self


class ScopedPolicyArguments(PolicyArguments):
    __doc__ = (PolicyArguments.__doc__ + " scope own (the default) keeps to your skills; use all when the case "
               "crosses into another area.")
    scope: Literal["own", "all"] = "own"


class SkillsPolicy:
    arguments = PolicyArguments
    description = PolicyArguments.__doc__
    prompt_hint = ("Use the available policy filenames to choose a matching skill (call policy with "
                   "action list if none are provided), then call policy with action open and that skill "
                   "to read its SKILL.md. Choose a clause named there and call policy with action read, "
                   "the same skill, and the exact clause filename. Only open reads SKILL.md; never call "
                   "read with SKILL.md. read requires both skill and a clause filename. After an error, "
                   "correct the arguments from its message instead "
                   "of repeating the same call. Never invent policy filenames or missing facts.")

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
        filename = "SKILL.md" if action == "open" else args.get("file", "")
        # Keep entrypoint access on open so read always refers to a clause file.
        if action == "read" and (
            filename.casefold() in {"policy.md", "skill.md", "index.md"} or filename == skill
        ):
            raise ValueError("read requires a clause filename listed in SKILL.md; use open with action and skill only to read SKILL.md")
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

    def has_skill(self, name: str) -> bool:
        """Whether `name` is a skill folder with a SKILL.md, without reading every skill."""
        base = (self.root / name).resolve()
        return base.parent == self.root and not (self.root / name).is_symlink() and (base / "SKILL.md").is_file()

    def scoped(self, scope: Sequence[str]) -> ScopedSkillsPolicy:
        """The same tool for a specialist: by default only its own skill folders."""
        if missing := [skill for skill in scope if not self.has_skill(skill)]:
            raise ValueError(f"policy scope names skills that do not exist: {missing}")
        return ScopedSkillsPolicy(self, tuple(scope))


class ScopedSkillsPolicy:
    """A specialist's view of the skills: list, open and read keep to `scope` unless the call says scope all."""
    arguments = ScopedPolicyArguments
    description = ScopedPolicyArguments.__doc__

    def __init__(self, policy: SkillsPolicy, scope: tuple[str, ...]):
        self.policy, self.scope = policy, scope
        self.prompt_hint = (policy.prompt_hint + " Your policy skills cover your own area; use scope all when the "
                            "case crosses into another area.")
        self.citable, self.warm_up = policy.citable, policy.warm_up

    def _own(self, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in catalog if item["skill"] in self.scope]

    def planner_context(self) -> str:
        # Only the specialist's own skills, so a plan does not list reads the default scope refuses.
        return ("Your policy skills (contents must still be read with tools; scope all reaches the others): "
                + json.dumps(self._own(self.policy.catalog())))

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        everything = args.get("scope", "own") == "all"
        action = args["action"]
        if action == "check":
            return self.policy.run(args)
        if action == "list":
            catalog = self.policy.catalog() if everything else self._own(self.policy.catalog())
            return {"skills": [item["skill"] for item in catalog], "catalog": catalog,
                    "scope": "all" if everything else "own"}
        skill = args.get("skill") or args.get("file", "")
        if not self.policy.has_skill(skill):
            # A guessed name: offer only names that exist, never another area's.
            choices = ([item["skill"] for item in self.policy.catalog()] if everything else list(self.scope))
            raise ValueError(f"no policy skill named {skill!r}; your skills: {choices}")
        if not everything and skill not in self.scope:
            raise ValueError(f"{skill} is outside your policy skills {list(self.scope)}; "
                             "use scope all if the case crosses into another area")
        return self.policy.run({key: value for key, value in args.items() if key != "scope"})
