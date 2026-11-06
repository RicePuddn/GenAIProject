"""Refresh the skills' clause files from the same PDFs used by RAG."""
import sys
from pathlib import Path

# Allow direct execution as ``python scripts/sync_skills.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SupportOpsBot.runtime.shared.config import ROOT
from SupportOpsBot.runtime.shared.policy.rag.loader import NUMBERED_HEADING, load_corpus, read_document


# Keep each read small enough for an agent to use one policy topic at a time.
RETURNS_FILES = {
    "1": "eligibility.md",
    "2": "eligibility.md",
    "3": "evidence.md",
    "4": "return-procedure.md",
    "5": "shipping-fees.md",
    "6": "refunds.md",
    "7": "disputes.md",
}
SHIPPING_FILES = {
    "1": "shipping-options.md",
    "2": "shipping-options.md",
    "3": "shipping-fees.md",
    "4": "changes.md",
    "5": "tracking.md",
    "6": "partial-deliveries.md",
    "7": "delivery-problems.md",
}


def sync_skills() -> None:
    files: dict[tuple[str, str], list[str]] = {}
    parent_titles = {}
    for source in sorted((ROOT / "data/policy/pdf").glob("*.pdf")):
        text, _ = read_document(source)
        for match in NUMBERED_HEADING.finditer(text):
            section_id, title = match.groups()
            if "." not in section_id:
                parent_titles[(source.name, section_id)] = title
    seen_parents: set[tuple[str, str]] = set()
    for clause in load_corpus(ROOT / "data/policy/pdf"):
        if clause.domain == "orders-and-shipping":
            filename = SHIPPING_FILES[clause.section_id.split(".")[0]]
        else:
            filename = RETURNS_FILES[clause.section_id.split(".")[0]]
        parent = clause.section_id.split(".")[0]
        parent_key = (clause.source, parent)
        body = ""
        if "." in clause.section_id and parent_key not in seen_parents:
            body += f"## {parent}. {parent_titles[parent_key]}\n\n"
        seen_parents.add(parent_key)
        level = "###" if "." in clause.section_id else "##"
        body += f"{level} {clause.id} {clause.title}\n\n{clause.text}\n"
        if clause.references:
            body += f"\nRelated clauses: {', '.join(clause.references)}.\n"
        files.setdefault((clause.domain, filename), []).append(body)
    for (domain, filename), sections in files.items():
        path = ROOT / "data/policy/skills" / domain / filename
        path.write_text("<!-- Generated from data/policy/pdf; run scripts/sync_skills.py to refresh. -->\n\n"
                        + "\n".join(sections))


if __name__ == "__main__":
    sync_skills()
    print("Updated skills clauses from the policy PDFs.")
