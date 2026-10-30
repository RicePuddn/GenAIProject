"""Refresh the skills' clause files from the same PDFs used by RAG."""
from support_ops.helpers.config import ROOT
from support_ops.policy.rag.loader import load_corpus


# Keep the existing skill filenames while copying every complete PDF clause.
def sync_policy_skills() -> None:
    files: dict[tuple[str, str], list[str]] = {}
    for clause in load_corpus(ROOT / "policy/raw_policy/pdf"):
        if clause.domain == "orders-and-shipping":
            filename = "changes.md" if clause.section_id.split(".")[0] == "4" else "tracking.md"
        else:
            filename = "refunds.md" if clause.section_id.split(".")[0] == "6" else "eligibility.md"
        body = f"## {clause.id} {clause.title}\n\n{clause.text}\n"
        if clause.references:
            body += f"\nRelated clauses: {', '.join(clause.references)}.\n"
        files.setdefault((clause.domain, filename), []).append(body)
    for (domain, filename), sections in files.items():
        path = ROOT / "policy/skills" / domain / filename
        path.write_text("<!-- Generated from policy/raw_policy/pdf; run scripts/sync_policy_skills.py to refresh. -->\n\n"
                        + "\n".join(sections))


if __name__ == "__main__":
    sync_policy_skills()
    print("Updated skills clauses from the policy PDFs.")
