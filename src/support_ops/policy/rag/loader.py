"""Policy files -> section chunks -> resolved references -> JSONL files.

Functions below follow this order:
1. read_document: read one PDF policy.
2. split_clauses: turn that policy's section bodies into chunks.
3. resolve_references: link section mentions after all chunks exist.
4. load_corpus: run steps 1-3 for every policy in a folder.
5. write_chunks: save the results as JSONL.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from support_ops.policy.rag.identifiers import POLICY_SECTION_PREFIXES

NUMBERED_HEADING = re.compile(r"^(?:#{1,6}\s+)?(\d+(?:\.\d+)*)(?:\.)?\s+([A-Z][^\n]+?)\s*$", re.MULTILINE)
SECTION_REFERENCE = re.compile(
    r"\bsections?\s+(\d+(?:\.\d+)*(?:\s*(?:,|and|or)\s*\d+(?:\.\d+)*)*)", re.IGNORECASE,
)


@dataclass(frozen=True)
class Clause:
    id: str
    title: str
    section: str
    source: str
    text: str
    references: tuple[str, ...] = ()
    domain: str = ""
    category: str = ""
    section_id: str = ""

    # Build retrieval text with the policy and section heading as context.
    def indexed_text(self) -> str:
        head = f"{self.section_id or self.id} {self.title}\n" if self.title else ""
        return f"{self.section}\n{head}{self.text}"


# Extract one PDF, removing repeated policy-title and page-number footers.
def read_document(path: Path) -> tuple[str, str]:
    if path.suffix.lower() != ".pdf":
        raise ValueError("RAG policy sources must be PDF files")
    import pypdf

    reader = pypdf.PdfReader(path)
    title = (reader.metadata.title if reader.metadata else None) or path.stem
    pages = []
    for number, page in enumerate(reader.pages, 1):
        lines = (page.extract_text() or "").splitlines()
        pages.append("\n".join(line for line in lines if line.strip() not in {title, str(number)}))
    text = "\n".join(pages)
    if not text.strip():
        raise ValueError(f"no extractable text in {path.name}; scanned PDFs need OCR first")
    return text, title


# Split at section headers, preserving the complete body below each header.
def split_clauses(text: str, section: str, source: str) -> list[Clause]:
    # Step 1: Find numbered section headings, such as 2 or 2.1.
    matches = list(NUMBERED_HEADING.finditer(text))
    if not matches:
        raise ValueError(f"no numbered policy headings found in {source}")
    seen = set()
    chunks = []
    category = ""
    previous = ""
    document_id = Path(source).stem
    domain = document_id.removesuffix("-policy")
    prefixes = POLICY_SECTION_PREFIXES.get(source, {})
    for match, following in zip(matches, [*matches[1:], None]):
        # Step 2: Check section IDs and their position under the correct parent.
        section_id, title = match.groups()
        if section_id in seen:
            raise ValueError(f"duplicate section {section_id} in {source}")
        parent = section_id.rpartition(".")[0]
        if parent and parent not in seen:
            raise ValueError(f"section {section_id} has no preceding parent section in {source}")
        if parent and not (previous == parent or previous.startswith(parent + ".")):
            raise ValueError(f"section {section_id} is outside its parent section in {source}")
        seen.add(section_id)
        previous = section_id
        if not parent:
            category = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        body = text[match.end():following.start() if following else len(text)].strip()
        if not body:
            if following and following.group(1).startswith(section_id + "."):
                continue  # A heading containing only subsections has no body to embed.
            raise ValueError(f"empty section {section_id} in {source}")
        # Step 3: Keep the entire section body, including its lists and exceptions.
        prefix = prefixes.get(section_id.split(".")[0], document_id)
        chunks.append(Clause(
            id=f"{prefix}-{section_id}", title=title, section=section,
            source=source, text=" ".join(body.split()), domain=domain,
            category=category, section_id=section_id,
        ))
    return chunks


# Resolve section mentions only after every chunk and its ID have been created.
def resolve_references(clauses: list[Clause]) -> list[Clause]:
    # Step 1: Validate IDs and prepare lookups for the completed corpus.
    ids = Counter(clause.id for clause in clauses)
    if duplicates := sorted(key for key, count in ids.items() if count > 1):
        raise ValueError(f"duplicate clause ids: {duplicates}")
    documents: dict[str, list[Clause]] = {}
    links = {clause.id: list(clause.references) for clause in clauses}
    for clause in clauses:
        documents.setdefault(clause.source, []).append(clause)

    # Step 2: Resolve explicit mentions within each policy; parent sections include their subsections.
    for clause in clauses:
        for match in SECTION_REFERENCE.finditer(clause.text):
            for section_id in re.findall(r"\d+(?:\.\d+)*", match.group(1)):
                targets = [target.id for target in documents[clause.source]
                           if target.section_id == section_id or target.section_id.startswith(section_id + ".")]
                if not targets:
                    raise ValueError(f"unknown section {section_id} referenced by {clause.id}")
                links[clause.id].extend(targets)

    # Step 3: Remove duplicate and self references, without following links recursively.
    return [replace(clause, references=tuple(dict.fromkeys(
        reference for reference in links[clause.id] if reference != clause.id
    ))) for clause in clauses]


# Load every policy in a folder, collect its chunks and validate the results.
def load_corpus(folder: Path) -> list[Clause]:
    # Step 1: Select the source PDFs.
    files = sorted(folder.glob("*.pdf"))
    if not files:
        raise ValueError(f"no PDF policy files in {folder}")
    # Step 2: Read each document and split only at section headings.
    clauses: list[Clause] = []
    for path in files:
        text, section = read_document(path)
        clauses.extend(split_clauses(text, section, path.name))
    # Step 3: Resolve and validate references after all chunks exist.
    return resolve_references(clauses)


# Save the validated chunks as JSONL files for the next pipeline stage.
def write_chunks(clauses: list[Clause], folder: Path) -> list[Path]:
    # Step 1: Group the section chunks by their source policy.
    grouped: dict[str, list[Clause]] = {}
    for clause in clauses:
        grouped.setdefault(Path(clause.source).stem, []).append(clause)
    # Step 2: Write one JSONL file per policy.
    paths = []
    folder.mkdir(parents=True, exist_ok=True)
    for document_id, chunks in grouped.items():
        path = folder / f"{document_id}.jsonl"
        records = [{
            "id": chunk.id, "source": chunk.source, "domain": chunk.domain,
            "category": chunk.category, "section_id": chunk.section_id,
            "section_title": chunk.title, "text": chunk.text, "references": list(chunk.references),
        } for chunk in chunks]
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="folder containing the policy PDFs")
    parser.add_argument("--output", type=Path, default=Path("policy/chunk_policy"))
    args = parser.parse_args()
    chunks = load_corpus(args.source)
    for path in write_chunks(chunks, args.output):
        print(path)
    print(f"Wrote {len(chunks)} section chunks.")
