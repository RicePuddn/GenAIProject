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
        chunks = split_clauses(text, "Returns Policy", "returns-policy.pdf")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, body)
        self.assertEqual(chunks[1].text, "Shipping fees apply.")
        self.assertEqual([c.section_id for c in chunks], ["1.1", "1.2"])
        self.assertEqual(chunks[0].id, "returns-policy-1.1")
        self.assertTrue(all(not hasattr(c, "chunk_index") and not hasattr(c, "chunk_count") for c in chunks))

    def test_parent_body_is_retained_separately_from_subsections(self):
        text = "## 1. Returns\nGeneral conditions apply.\n### 1.1 Eligibility\nReturn within thirty days."
        chunks = split_clauses(text, "Returns Policy", "returns-policy.pdf")
        self.assertEqual([c.section_id for c in chunks], ["1", "1.1"])
        self.assertEqual([c.text for c in chunks], ["General conditions apply.", "Return within thirty days."])
        self.assertEqual(chunks[0].category, chunks[1].category)
        self.assertNotIn("#", chunks[0].id)

    def test_invalid_sections_fail_clearly(self):
        for text in ("No numbered headings.", "## 1. Returns\n", "## 1.1 Missing parent\nText.",
                     "## 1. Returns\nText.\n## 2. Refunds\nText.\n### 1.1 Misplaced\nText.",
                     "## 1. Returns\nText.\n## 1. Duplicate\nText."):
            with self.subTest(text=text), self.assertRaises(ValueError):
                split_clauses(text, "Returns", "returns.pdf")

    def test_pdf_sources_produce_complete_serializable_chunks(self):
        chunks = load_corpus(ROOT / "policy/raw_policy/pdf")
        self.assertEqual(len(chunks), 52)
        self.assertTrue(all(c.source.endswith(".pdf") and c.text and c.domain for c in chunks))
        ids = {c.id for c in chunks}
        self.assertIn("ODR-OrderChangeCancel-4.1", ids)
        self.assertIn("SHP-SeparateShipPartialDel-6.1", ids)
        self.assertTrue(all(set(c.references) <= ids for c in chunks))
        cases = json.loads((ROOT / "data/cases/rag/questions.json").read_text())
        self.assertTrue(cases)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertTrue(all(set(case["expected_chunk_ids"]) <= ids for case in cases))
        by_id = {c.id: c for c in chunks}
        for case in cases:
            direct_references = {ref for key in case["expected_chunk_ids"] for ref in by_id[key].references}
            self.assertLessEqual(set(case.get("expected_reference_chunk_ids", [])), direct_references, case["id"])
        with tempfile.TemporaryDirectory() as directory:
            paths = write_chunks(chunks, Path(directory))
            records = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
            self.assertEqual(len(paths), 2)
            self.assertEqual([r["id"] for r in records], [c.id for c in chunks])
            self.assertEqual([r["references"] for r in records], [list(c.references) for c in chunks])

    def test_non_pdf_sources_and_empty_folders_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "policy.md").write_text("## 1. Returns\nText.")
            with self.assertRaisesRegex(ValueError, "PDF"):
                read_document(root / "policy.md")
            with self.assertRaisesRegex(ValueError, "no PDF"):
                load_corpus(root)


class ReferenceTests(unittest.TestCase):
    def test_references_resolve_after_chunk_creation_and_expand_parent_sections(self):
        text = ("## 1. Requests\nSee sections 2.1 and 2.2, then section 2.1 again.\n"
                "## 2. Evidence\n### 2.1 Photos\nSee section 3.\n"
                "### 2.2 Exceptions\nHuman review.\n## 3. Review\nExplain the reason.")
        clauses = split_clauses(text, "Policy", "policy.pdf")
        self.assertEqual(clauses[0].references, ())
        clauses = resolve_references(clauses)
        self.assertEqual(clauses[0].references, ("policy-2.1", "policy-2.2"))
        # The reference from 2.1 to 3 is not recursively copied into section 1.
        self.assertEqual(clauses[1].references, ("policy-3",))
        parent = split_clauses("## 1. Request\nSee section 2.\n## 2. Evidence\n"
                               "### 2.1 Photos\nPhotos.\n### 2.2 Exceptions\nHuman review.", "Policy", "policy.pdf")
        self.assertEqual(resolve_references(parent)[0].references, ("policy-2.1", "policy-2.2"))

    def test_unknown_references_and_duplicate_ids_are_rejected(self):
        clauses = split_clauses("## 1. Requests\nSee section 9.1.", "Policy", "policy.pdf")
        with self.assertRaisesRegex(ValueError, "unknown section 9.1"):
            resolve_references(clauses)
        with self.assertRaisesRegex(ValueError, "duplicate clause ids"):
            resolve_references(clauses + clauses)

    def test_references_do_not_cross_between_policies(self):
        first = split_clauses("## 1. Requests\nSee section 2.", "First", "first.pdf")
        second = split_clauses("## 2. Evidence\nPhotos.", "Second", "second.pdf")
        with self.assertRaisesRegex(ValueError, "unknown section 2"):
            resolve_references(first + second)

    def test_evidence_rule_and_exception_reference_each_other(self):
        clauses = load_corpus(ROOT / "policy/raw_policy/pdf")
        by_id = {clause.id: clause for clause in clauses}
        rule = "RET-SupportingEvidence-3.1"
        exception = "RET-SupportingEvidence-3.2"
        self.assertIn(exception, by_id[rule].references)
        self.assertIn(rule, by_id[exception].references)
