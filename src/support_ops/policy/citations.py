"""A proposal may cite only clauses or files its policy tool returned in this ticket.

Shared by the single-agent `backend_write` tool and the multi-agent specialists' `route` tool, so
both architectures hold citations to the same rule. Each provider's `citable` says what a policy
result lets the agent cite (RAG: clause IDs; skills: the file read and its clause headings).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def citation_forms(clause: str) -> set[str]:
    """The ways a citation may name what was read, casefolded: as written, each side of a `#` anchor, a
    path cut to its last two parts (skill/file, so another skill's file of the same name does not match),
    and the leading clause ID of "RET-1 Return window"."""
    forms = set()
    for part in clause.strip().casefold().split("#"):
        part = part.strip().strip("/")
        if not part:
            continue
        pieces = part.split("/")
        forms |= {part, "/".join(pieces[-2:]), part.split()[0]}
    return forms


def unread_citations(cited: Iterable[Any], citable: Iterable[str]) -> list[str]:
    """The cited clauses that match nothing the policy tool returned."""
    known = {item.casefold() for item in citable}
    return [str(clause) for clause in cited if not citation_forms(str(clause)) & known]
