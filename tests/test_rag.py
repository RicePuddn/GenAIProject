"""RAG policy retrieval (S1): loading, chunking, each retrieval stage, the tool and the workflow.

Stand-in models keep these tests offline: a bag-of-words embedder and a word-overlap reranker.
"""
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from support_ops.backend.postgres import PostgresReader
from support_ops.gateway.ticket_state import TicketState
from support_ops.helpers.clock import Clock
from support_ops.helpers.config import ROOT, load_config
from support_ops.helpers.llm import ScriptedLLM, text, tool_call
from support_ops.policy.rag import RagPolicy
from support_ops.policy.rag.loader import PDF_CLAUSE, load_corpus, split_clauses, split_windows
from support_ops.policy.rag.retriever import BM25, PolicyRetriever, RagSettings, reciprocal_rank_fusion
from support_ops.systems.single_agent import SingleAgentWorkflow
from support_ops.systems.tools import SupportTools

MINI = ROOT / "policy/corpus/mini"
ALL_STAGES = RagSettings()


class WordEmbedder:
    """Counts of a fixed vocabulary: similar wording gives similar vectors."""
    VOCAB = ["return", "returns", "refund", "refunds", "days", "delivery", "unused", "final", "sale",
             "tracking", "shipment", "delayed", "order", "change", "cancel", "payment", "approval"]

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return [[t.lower().split().count(word) + t.lower().count(word) * 0.01 for word in self.VOCAB]
                for t in texts]


class OverlapReranker:
    """Scores a passage by how many distinct query words it contains."""

    def __init__(self, boost: str | None = None):
        self.boost, self.calls = boost, 0

    def score(self, query, passages):
        self.calls += 1
        words = set(query.lower().split())
        return [len(words & set(p.lower().split())) + (100 if self.boost and self.boost in p else 0)
                for p in passages]


def retriever(settings=ALL_STAGES, embedder=None, reranker=None):
    return PolicyRetriever(load_corpus(MINI, settings.chunking), settings,
                           embedder or WordEmbedder(), reranker or OverlapReranker())


class LoadingTests(unittest.TestCase):
    def test_mini_corpus_has_clause_ids_sections_and_exception_links(self):
        clauses = {clause.id: clause for clause in load_corpus(MINI)}
        self.assertEqual(set(clauses), {"RET-1", "RET-1-EX-1", "REF-1", "SHP-1", "ORD-1"})
        self.assertEqual(clauses["RET-1-EX-1"].exception_of, ("RET-1",))
        self.assertEqual(clauses["RET-1"].section, "Returns and refunds")
        self.assertEqual(clauses["RET-1"].source, "returns-and-refunds.md")
        self.assertNotIn("Exception to", clauses["RET-1-EX-1"].text)
        self.assertTrue(clauses["RET-1"].text.startswith("Customers may request a return within 30 days"))

    def test_pdf_text_without_headings_splits_on_clause_id_lines(self):
        extracted = ("Returns policy\nRET-1 Return window\nReturns within 30 days.\n"
                     "RET-1-EX-1 Final-sale items\nException to: RET-1\nFinal-sale items cannot be returned.\n")
        clauses = split_clauses(extracted, "returns", "returns.pdf", PDF_CLAUSE)
        self.assertEqual([(c.id, c.title) for c in clauses],
                         [("RET-1", "Return window"), ("RET-1-EX-1", "Final-sale items")])
        self.assertEqual(clauses[1].exception_of, ("RET-1",))
        self.assertEqual(clauses[0].text, "Returns within 30 days.")

    def test_window_chunking_overlaps_and_covers_every_word(self):
        words = " ".join(f"w{i}" for i in range(25))
        chunks = split_windows(words, "s", "doc.md", size=10, overlap=3)
        self.assertEqual([c.id for c in chunks], ["doc#w1", "doc#w2", "doc#w3", "doc#w4"])
        self.assertEqual(chunks[1].text.split()[0], "w7")
        self.assertEqual(chunks[-1].text.split()[-1], "w24")
        self.assertEqual(len(load_corpus(MINI, chunking="window")), 2)

    def test_bad_corpora_fail_loudly(self):
        cases = {
            "no ids": "# Title\nJust prose.\n",
            "duplicate": "## RET-1 A\nx\n## RET-1 B\ny\n",
            "dangling": "## RET-1 A\nx\n## RET-2 B\nException to: RET-9\ny\n",
        }
        for name, content in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                Path(directory, "policy.md").write_text(content)
                with self.assertRaises(ValueError):
                    load_corpus(Path(directory))

    def test_invalid_settings_are_rejected(self):
        for settings in ({"dense": False, "bm25": False}, {"k": 30}, {"chunking": "pages"}, {"typo": 1}):
            with self.subTest(settings), self.assertRaises(ValueError):
                RagSettings.from_config(settings)


class RetrievalStageTests(unittest.TestCase):
    def test_bm25_ranks_the_clause_with_the_exact_term_first(self):
        clauses = load_corpus(MINI)
        scores = BM25([c.indexed_text(False) for c in clauses]).scores("final-sale")
        self.assertEqual(clauses[scores.index(max(scores))].id, "RET-1-EX-1")
        scores = BM25([c.indexed_text(False) for c in clauses]).scores("ORD-1")
        self.assertEqual(clauses[scores.index(max(scores))].id, "ORD-1")

    def test_rank_fusion_rewards_agreement_between_lists(self):
        fused = reciprocal_rank_fusion([[0, 1, 2], [1, 2, 0]], k=60)
        self.assertEqual(max(fused, key=fused.get), 1)
        self.assertAlmostEqual(fused[1], 1 / 62 + 1 / 61)

    def test_each_stage_can_be_switched_off(self):
        embedder, reranker = WordEmbedder(), OverlapReranker()
        keyword_only = replace(ALL_STAGES, dense=False, rerank=False)
        hits = retriever(keyword_only, embedder, reranker).search("final-sale")
        self.assertEqual(hits[0]["id"], "RET-1-EX-1")
        self.assertEqual((embedder.calls, reranker.calls), (0, 0))

        hits = retriever(replace(ALL_STAGES, bm25=False, rerank=False), embedder).search("tracking delayed shipment")
        self.assertEqual(hits[0]["id"], "SHP-1")
        self.assertEqual(embedder.calls, 2)   # once for the corpus, once for the query

    def test_keyword_search_returns_nothing_when_no_term_matches(self):
        settings = replace(ALL_STAGES, dense=False, rerank=False)
        self.assertEqual(retriever(settings).search("zzzz qqqq"), [])

    def test_reranker_reorders_the_candidates_and_its_score_is_reported(self):
        settings = replace(ALL_STAGES, expand_exceptions=False)
        hits = retriever(settings, reranker=OverlapReranker(boost="Order changes")).search("refund approval")
        self.assertEqual(hits[0]["id"], "ORD-1")
        self.assertGreaterEqual(hits[0]["score"], 100)

    def test_corpus_is_embedded_once_across_searches(self):
        embedder = WordEmbedder()
        search = retriever(embedder=embedder)
        search.search("refund")
        search.search("tracking")
        self.assertEqual(embedder.calls, 3)

    def test_warm_up_loads_models_and_embeds_the_corpus_before_the_first_search(self):
        embedder, reranker = WordEmbedder(), OverlapReranker()
        search = retriever(embedder=embedder, reranker=reranker)
        search.warm_up()
        self.assertIsNotNone(search._vectors)
        self.assertEqual((embedder.calls, reranker.calls), (2, 1))
        search.search("refund")
        self.assertEqual((embedder.calls, reranker.calls), (3, 2))   # only the query is embedded

        keyword_only = retriever(replace(ALL_STAGES, dense=False, rerank=False))
        with patch("support_ops.policy.rag.retriever.load_embedder") as load:
            keyword_only.warm_up()
        load.assert_not_called()

    def test_returned_clauses_bring_their_exceptions_once(self):
        settings = replace(ALL_STAGES, k=1)
        hits = retriever(settings).search("return within 30 days of delivery unused")
        self.assertEqual([(h["id"], h["via"]) for h in hits],
                         [("RET-1", "search"), ("RET-1-EX-1", "exception_of:RET-1")])
        self.assertIsNone(hits[1]["score"])

        hits = retriever(replace(ALL_STAGES, k=5)).search("return final sale")
        self.assertEqual(sorted(h["id"] for h in hits), ["ORD-1", "REF-1", "RET-1", "RET-1-EX-1", "SHP-1"])
        self.assertNotIn("exception_of:RET-1", [h["via"] for h in hits])

        hits = retriever(replace(ALL_STAGES, k=1, expand_exceptions=False)).search("return within 30 days")
        self.assertEqual([h["id"] for h in hits], ["RET-1"])

    def test_contextual_chunks_prepend_the_section(self):
        clause = load_corpus(MINI)[0]
        self.assertTrue(clause.indexed_text(True).startswith(f"{clause.section}\n{clause.id} {clause.title}"))
        self.assertTrue(clause.indexed_text(False).startswith(f"{clause.id} {clause.title}"))


class RagToolTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock(datetime(2026, 10, 1, tzinfo=timezone.utc))
        self.policy = RagPolicy(retriever())

    def test_tool_contract_requires_query_for_search_and_amount_for_refund_check(self):
        tools = SupportTools(Mock(spec=PostgresReader), self.policy, TicketState(self.clock))
        for args in ({"action": "search"}, {"action": "list"}, {"action": "check", "operation": "refund"},
                     {"action": "search", "query": ""}, {"action": "search", "query": "x", "skill": "y"}):
            with self.subTest(args), self.assertRaises(ValueError):
                tools.validate("policy", args)
        self.assertEqual(tools.validate("policy", {"action": "search", "query": "refund"}),
                         {"action": "search", "query": "refund"})

    def test_search_and_shared_rule_check(self):
        result = self.policy.run({"action": "search", "query": "refund approval limit"})
        self.assertEqual(result["query"], "refund approval limit")
        self.assertIn("REF-1", [clause["id"] for clause in result["clauses"]])
        self.assertEqual(self.policy.run({"action": "check", "operation": "refund", "amount": 150})["requires_human"], True)

    def test_planner_sees_section_names_only(self):
        context = self.policy.planner_context()
        self.assertIn('"Orders and shipping"', context)
        self.assertNotIn("30 days", context)


class RagWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.cfg["policy"]["mode"] = "rag"
        self.reader = Mock(spec=PostgresReader)

    def test_rag_mode_builds_without_loading_models(self):
        with patch("support_ops.policy.rag.retriever.load_embedder") as embedder, \
             patch("support_ops.policy.rag.retriever.load_reranker") as reranker:
            workflow = SingleAgentWorkflow(ScriptedLLM(), self.cfg, reader=self.reader)
        self.assertIsInstance(workflow.tools.policy, RagPolicy)
        embedder.assert_not_called()
        reranker.assert_not_called()

    def test_api_warms_up_the_configured_policy_once_at_startup(self):
        from fastapi.testclient import TestClient
        from support_ops.api import app

        with patch("support_ops.api.policy_from_config") as build, TestClient(app):
            build.return_value.warm_up.assert_called_once()
        build.return_value.warm_up.assert_called_once()

    def test_react_agent_searches_the_policy_and_answers(self):
        llm = ScriptedLLM().push("guard", text("0.01")).push(
            "supervisor", tool_call("policy", action="search", query="final sale return"),
            text("ready"), text("Final-sale items cannot be returned (RET-1-EX-1)."),
        )
        workflow = SingleAgentWorkflow(llm, self.cfg, reader=self.reader, policy=RagPolicy(retriever()))
        result = workflow.invoke("T1", "Can I return a final-sale item?", "react")
        ids = [clause["id"] for clause in result["evidence"][0]["result"]["clauses"]]
        self.assertIn("RET-1-EX-1", ids)
        self.assertIn("RET-1", ids)
        system = llm.calls[1][1][0]["content"]
        self.assertIn("Search the policy for the relevant clauses", system)
        self.assertNotIn("SKILL.md", system)
        self.assertIn("RET-1-EX-1", str(result["trace"].tool_calls[0]["execution"]["output"]))


@unittest.skipUnless(importlib.util.find_spec("sentence_transformers"), "rag extra not installed")
class RealModelTests(unittest.TestCase):
    def test_real_models_find_the_exception_for_a_paraphrased_question(self):
        hits = PolicyRetriever(load_corpus(MINI), ALL_STAGES).search(
            "can I send back something I bought on clearance")
        self.assertIn("RET-1-EX-1", [hit["id"] for hit in hits])


if __name__ == "__main__":
    unittest.main()
