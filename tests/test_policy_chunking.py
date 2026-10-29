"""Section-based policy loading: boundaries, validation and reproducible output."""
import json
import tempfile
import unittest
from pathlib import Path

from support_ops.policy.rag.loader import (
    resolve_references, load_corpus, read_document, split_clauses, write_chunks,
)

ROOT = Path(__file__).resolve().parents[1]

class PolicyChunkingTests(unittest.TestCase):
    def test_long_section_stays_whole_with_its_list_and_exception(self):
        body = " ".join(f"Condition {i} applies to the affected goods." for i in range(150))
        body += " Required contents: (i) goods; (ii) accessories. Exception: unsafe evidence requires human review."
        text = "## 1. Returns\n### 1.1 Eligibility\n" + body + "\n### 1.2 Fees\nShipping fees apply."
        chunks = split_clauses(text, "Returns Policy", "returns-policy.md")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, body)
        self.assertEqual(chunks[1].text, "Shipping fees apply.")
        self.assertEqual([c.section_id for c in chunks], ["1.1", "1.2"])
        self.assertEqual(chunks[0].id, "returns-policy-1.1")
        self.assertTrue(all(not hasattr(c, "chunk_index") and not hasattr(c, "chunk_count") for c in chunks))

    def test_parent_body_is_retained_separately_from_subsections(self):
        text = "## 1. Returns\nGeneral conditions apply.\n### 1.1 Eligibility\nReturn within thirty days."
        chunks = split_clauses(text, "Returns Policy", "returns-policy.md")
        self.assertEqual([c.section_id for c in chunks], ["1", "1.1"])
        self.assertEqual([c.text for c in chunks], ["General conditions apply.", "Return within thirty days."])
        self.assertEqual(chunks[0].category, chunks[1].category)
        self.assertNotIn("#", chunks[0].id)

    def test_invalid_sections_fail_clearly(self):
        for text in ("No numbered headings.", "## 1. Returns\n", "## 1.1 Missing parent\nText.",
                     "## 1. Returns\nText.\n## 2. Refunds\nText.\n### 1.1 Misplaced\nText.",
                     "## 1. Returns\nText.\n## 1. Duplicate\nText."):
            with self.subTest(text=text), self.assertRaises(ValueError):
                split_clauses(text, "Returns", "returns.md")

    def test_pdf_and_markdown_produce_the_same_clause_bodies(self):
        root = ROOT / "policy/raw_policy"
        markdown = load_corpus(root / "markdown")
        pdf = load_corpus(root / "pdf")
        self.assertEqual([(c.id, c.text, c.category) for c in markdown],
                         [(c.id, c.text, c.category) for c in pdf])
        self.assertEqual([c.references for c in markdown], [c.references for c in pdf])
        self.assertGreaterEqual(len(pdf), 49)
        self.assertTrue(all(c.domain and c.category and c.section_id for c in pdf))
        with tempfile.TemporaryDirectory() as directory:
            paths = write_chunks(pdf, Path(directory))
            records = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
            self.assertEqual(len(paths), 2)
            self.assertEqual([record["id"] for record in records], [c.id for c in pdf])
            self.assertTrue(all("section_title" in record for record in records))
            self.assertEqual([record["references"] for record in records],
                             [list(c.references) for c in pdf])

    def test_duplicate_formats_and_invalid_frontmatter_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "policy.md").write_text("## 1. Returns\nText.")
            (root / "policy.pdf").touch()
            with self.assertRaisesRegex(ValueError, "both copies"):
                load_corpus(root)
            (root / "policy.md").write_text("---\n- wrong\n---\n# Returns")
            with self.assertRaisesRegex(ValueError, "mapping"):
                read_document(root / "policy.md")


class ReferenceTests(unittest.TestCase):
    def test_references_resolve_after_chunk_creation_and_expand_parent_sections(self):
        text = ("## 1. Requests\nSee sections 2.1 and 2.2, then section 2.1 again.\n"
                "## 2. Evidence\n### 2.1 Photos\nSee section 3.\n"
                "### 2.2 Exceptions\nHuman review.\n## 3. Review\nExplain the reason.")
        clauses = split_clauses(text, "Policy", "policy.md")
        self.assertEqual(clauses[0].references, ())
        clauses = resolve_references(clauses)
        self.assertEqual(clauses[0].references, ("policy-2.1", "policy-2.2"))
        # The reference from 2.1 to 3 is not recursively copied into section 1.
        self.assertEqual(clauses[1].references, ("policy-3",))
        parent = split_clauses("## 1. Request\nSee section 2.\n## 2. Evidence\n"
                               "### 2.1 Photos\nPhotos.\n### 2.2 Exceptions\nHuman review.", "Policy", "policy.md")
        self.assertEqual(resolve_references(parent)[0].references, ("policy-2.1", "policy-2.2"))

    def test_unknown_references_and_duplicate_ids_are_rejected(self):
        clauses = split_clauses("## 1. Requests\nSee section 9.1.", "Policy", "policy.md")
        with self.assertRaisesRegex(ValueError, "unknown section 9.1"):
            resolve_references(clauses)
        with self.assertRaisesRegex(ValueError, "duplicate clause ids"):
            resolve_references(clauses + clauses)

    def test_references_do_not_cross_between_policies(self):
        first = split_clauses("## 1. Requests\nSee section 2.", "First", "first.md")
        second = split_clauses("## 2. Evidence\nPhotos.", "Second", "second.md")
        with self.assertRaisesRegex(ValueError, "unknown section 2"):
            resolve_references(first + second)

    def test_evidence_rule_and_exception_reference_each_other(self):
        clauses = load_corpus(ROOT / "policy/raw_policy/pdf")
        by_id = {clause.id: clause for clause in clauses}
        rule = "returns-and-refunds-policy-3.1"
        exception = "returns-and-refunds-policy-3.2"
        self.assertIn(exception, by_id[rule].references)
        self.assertIn(rule, by_id[exception].references)
