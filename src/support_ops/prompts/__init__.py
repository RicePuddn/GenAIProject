"""Every role's prompt, as Markdown files beside this module, so they can be read and compared across setups.

A file is one prompt, or several named sections each under a `## name` heading (text before the first
heading is a note for readers and is never sent). `{name}` marks a value filled in by code; every
placeholder must be given and every value used, so a renamed placeholder fails loudly.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
HEADING = re.compile(r"^## ([a-z_]+)[ \t]*$", re.MULTILINE)


@lru_cache(maxsize=None)
def _sections(name: str) -> dict[str | None, str]:
    text = (FOLDER / f"{name}.md").read_text(encoding="utf-8")
    parts = HEADING.split(text)
    if len(parts) == 1:
        return {None: text}
    return {heading: body for heading, body in zip(parts[1::2], parts[2::2])}


def load(name: str, section: str | None = None, **values: object) -> str:
    """The prompt `name` (its `section`, for a file of sections) with its placeholders filled in."""
    sections = _sections(name)
    if section not in sections:
        raise KeyError(f"prompts/{name}.md has no section {section!r}")
    template = sections[section]
    wanted = set(PLACEHOLDER.findall(template))
    if missing := wanted - set(values):
        raise ValueError(f"prompts/{name}.md {section or ''} needs {sorted(missing)}")
    if unused := set(values) - wanted:
        raise ValueError(f"prompts/{name}.md {section or ''} has no placeholder for {sorted(unused)}")
    return PLACEHOLDER.sub(lambda match: str(values[match[1]]), template).strip()
