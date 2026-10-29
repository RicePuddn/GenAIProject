"""Turn policy files (.md or .pdf) into clause chunks. Runs once at startup.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

CLAUSE_ID = r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-\d+"
# Markdown clauses start with a heading; PDF text has no headings, so a line starting with an id.
MARKDOWN_CLAUSE = re.compile(rf"^#{{1,6}}\s+({CLAUSE_ID})\s+(\S.*?)\s*$", re.MULTILINE)
PDF_CLAUSE = re.compile(rf"^\s*({CLAUSE_ID})\s+(\S.*?)\s*$", re.MULTILINE)
EXCEPTION_LINE = re.compile(rf"^\s*Exception to:\s*({CLAUSE_ID}(?:\s*,\s*{CLAUSE_ID})*)\s*$", re.MULTILINE)
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
TITLE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Clause:
    id: str
    title: str
    section: str
    source: str
    text: str
    exception_of: tuple[str, ...] = ()

    def indexed_text(self, contextual: bool) -> str:
        """The text that gets searched; contextual prepends the section (Topic 8 contextual RAG)."""
        head = f"{self.id} {self.title}\n" if self.title else ""
        return f"{self.section}\n{head}{self.text}" if contextual else f"{head}{self.text}"


def read_document(path: Path) -> tuple[str, str]:
    """Return a document's text and its section name."""
    if path.suffix == ".pdf":
        try:
            import pypdf
        except ImportError as exc:
            raise RuntimeError("reading PDF policies needs the rag extra: pip install -e '.[rag]'") from exc
        return "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(path).pages), path.stem
    text = path.read_text(encoding="utf-8")
    meta = {}
    if match := FRONTMATTER.match(text):
        meta = yaml.safe_load(match.group(1)) or {}
        text = text[match.end():]
    title = TITLE.search(text)
    return text, str(meta.get("section") or (title.group(1) if title else path.stem))


def split_clauses(text: str, section: str, source: str, pattern: re.Pattern = MARKDOWN_CLAUSE) -> list[Clause]:
    """One chunk per clause: from its id line to the next. Text before the first clause is skipped."""
    matches = list(pattern.finditer(text))
    clauses = []
    for match, following in zip(matches, [*matches[1:], None]):
        body = text[match.end():following.start() if following else len(text)]
        links = tuple(clause_id for line in EXCEPTION_LINE.finditer(body)
                      for clause_id in re.split(r"\s*,\s*", line.group(1)))
        clauses.append(Clause(match.group(1), match.group(2), section, source,
                              " ".join(EXCEPTION_LINE.sub("", body).split()), links))
    return clauses


def split_windows(text: str, section: str, source: str, size: int, overlap: int) -> list[Clause]:
    """Overlapping word windows, as in Lab B's split_into_chunks."""
    words, step, chunks = text.split(), size - overlap, []
    for start in range(0, len(words), step):
        chunks.append(Clause(f"{Path(source).stem}#w{len(chunks) + 1}", "", section, source,
                             " ".join(words[start:start + size])))
        if start + size >= len(words):
            break
    return chunks


def load_corpus(folder: Path, chunking: str = "clause", window_words: int = 180,
                window_overlap: int = 30) -> list[Clause]:
    files = sorted(path for path in folder.glob("*") if path.suffix in {".md", ".pdf"}
                   and path.name.lower() != "readme.md")
    if not files:
        raise ValueError(f"no .md or .pdf policy files in {folder}")
    clauses: list[Clause] = []
    for path in files:
        text, section = read_document(path)
        if chunking == "window":
            clauses += split_windows(text, section, path.name, window_words, window_overlap)
            continue
        found = split_clauses(text, section, path.name, PDF_CLAUSE if path.suffix == ".pdf" else MARKDOWN_CLAUSE)
        if not found:
            raise ValueError(f"no clause ids found in {path.name}; fix the ids or use policy.rag.chunking: window")
        clauses += found
    ids = Counter(clause.id for clause in clauses)
    if duplicates := sorted(clause_id for clause_id, count in ids.items() if count > 1):
        raise ValueError(f"duplicate clause ids in {folder}: {duplicates}")
    if dangling := sorted({link for clause in clauses for link in clause.exception_of} - set(ids)):
        raise ValueError(f"exceptions point to unknown clause ids: {dangling}")
    return clauses
