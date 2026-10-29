"""Offline RAG checks: saved embeddings, retrieval, evaluation metrics and agent tools."""
import tempfile
import unittest
import nbformat
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from pinecone import Document
from unittest.mock import Mock, patch

from support_ops.backend.postgres import PostgresReader
from support_ops.gateway.ticket_state import TicketState
from support_ops.helpers.clock import Clock
from support_ops.helpers.config import load_config
from support_ops.helpers.llm import ScriptedLLM, text, tool_call
from support_ops.policy.rag import RagPolicy
from support_ops.policy.rag.embeddings import DIMENSION, prepare_embeddings, read_embeddings
from support_ops.policy.rag.loader import EMBEDDING_MODEL
from support_ops.policy.rag.retriever import PineconeRetriever, RagSettings, reciprocal_rank_fusion, retrieve
from support_ops.systems.single_agent import SingleAgentWorkflow
from support_ops.systems.tools import SupportTools


class RetrievalEntryTests(unittest.TestCase):
    def test_rank_fusion_rewards_agreement_between_lists(self):
        scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
        self.assertEqual(max(scores, key=scores.get), "b")
        self.assertAlmostEqual(scores["b"], 1 / 62 + 1 / 61)

    def test_public_entry_point_uses_the_organisation_configuration(self):
        with patch("support_ops.policy.rag.retriever.shared_retriever") as build:
            build.return_value.search.return_value = [{"id": "returns-and-refunds-policy-3.2"}]
            result = retrieve("Evidence is unsafe")
            settings = build.call_args.args[0]
            self.assertEqual(settings, RagSettings.from_config(load_config()["policy"]["rag"]))
            build.return_value.search.assert_called_once_with("Evidence is unsafe")
            self.assertEqual(result[0]["id"], "returns-and-refunds-policy-3.2")

    def test_invalid_settings_fail(self):
        for values in ({"dense": False, "bm25": False}, {"k": 30}, {"typo": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RagSettings.from_config(values)


class RagToolTests(unittest.TestCase):
    def setUp(self):
        self.engine = Mock()
        self.engine.sections.return_value = ["Orders and Shipping Policy", "Returns and Refunds Policy"]
        self.engine.search.return_value = [{"id": "returns-and-refunds-policy-2.5", "text": "Faulty final-sale goods may qualify."}]
        self.policy = RagPolicy(self.engine)
        self.clock = Clock(datetime(2026, 10, 1, tzinfo=timezone.utc))

    def test_tool_validates_arguments_and_uses_the_shared_search(self):
        tools = SupportTools(Mock(spec=PostgresReader), self.policy, TicketState(self.clock))
        for args in ({"action": "search"}, {"action": "list"}, {"action": "check", "operation": "refund"},
                     {"action": "search", "query": ""}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                tools.validate("policy", args)
        result = self.policy.run({"action": "search", "query": "damaged final-sale item"})
        self.engine.search.assert_called_once_with("damaged final-sale item")
        self.assertEqual(result["clauses"], self.engine.search.return_value)
        self.assertTrue(self.policy.run({"action": "check", "operation": "refund", "amount": 150})["requires_human"])

    def test_planner_sees_policy_titles_only(self):
        context = self.policy.planner_context()
        self.assertIn("Orders and Shipping Policy", context)
        self.assertNotIn("Faulty final-sale", context)

    def test_workflow_builds_with_the_shared_retriever(self):
        cfg = load_config()
        with patch("support_ops.policy.rag.policy.shared_retriever", return_value=self.engine) as build:
            workflow = SingleAgentWorkflow(ScriptedLLM(), cfg, reader=Mock(spec=PostgresReader))
        self.assertIs(workflow.tools.policy.retriever, self.engine)
        build.assert_called_once()

    def test_react_workflow_returns_the_policy_evidence(self):
        llm = ScriptedLLM().push("guard", text("0.01")).push(
            "supervisor", tool_call("policy", action="search", query="damaged final-sale item"),
            text("ready"), text("Faulty final-sale goods may qualify (section 2.5)."),
        )
        workflow = SingleAgentWorkflow(llm, load_config(), reader=Mock(spec=PostgresReader), policy=self.policy)
        result = workflow.invoke("T1", "My final-sale item is faulty", "react")
        self.assertEqual(result["evidence"][0]["result"]["clauses"], self.engine.search.return_value)
        self.assertIn("Search the policy for the relevant clauses", llm.calls[1][1][0]["content"])


class SavedEmbeddingTests(unittest.TestCase):
    def test_embedding_cache_is_reused_only_for_the_same_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "embeddings"
            source.mkdir()
            (source / "returns-policy.md").write_text("# Returns Policy\n## 1. Returns\nReturns are accepted.")
            embedder = Mock()
            embedder.encode.return_value = [[1.0] + [0.0] * (DIMENSION - 1)]
            with patch("support_ops.policy.rag.embeddings.ROOT", root):
                first = prepare_embeddings(source, output, embedder)
                second = prepare_embeddings(source, output, embedder)
                self.assertEqual(embedder.encode.call_count, 1)
                self.assertEqual(first, second)
                manifest, documents = read_embeddings(output)
                self.assertEqual(len(documents[0]["embedding"]), DIMENSION)
                self.assertEqual(manifest["embedding_model"], EMBEDDING_MODEL)
                self.assertNotIn("embedding_model", documents[0])
                self.assertNotIn("corpus_id", documents[0])
                self.assertNotIn("text", documents[0])
                self.assertTrue(embedder.encode.call_args.args[0][0].startswith("search_document: Returns Policy"))
                (source / "returns-policy.md").write_text("# Returns Policy\n## 1. Returns\nReturns require review.")
                changed = prepare_embeddings(source, output, embedder)
                self.assertNotEqual(first["corpus_hash"], changed["corpus_hash"])
                self.assertEqual(embedder.encode.call_count, 2)
                (output / "policies.jsonl").write_text("tampered")
                with self.assertRaisesRegex(ValueError, "manifest"):
                    read_embeddings(output)

    def test_invalid_vector_dimensions_do_not_produce_uploadable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            (root / "source/policy.md").write_text("## 1. Returns\nReturn goods.")
            with patch("support_ops.policy.rag.embeddings.ROOT", root), self.assertRaisesRegex(ValueError, "768"):
                prepare_embeddings(root / "source", root / "embeddings", Mock(encode=Mock(return_value=[[1.0]])))
            self.assertFalse((root / "embeddings/manifest.json").exists())


def document(key, score=1.0, references=None):
    return Document({"_id": key, "_score": score, "section_title": key, "search_text": f"Policy {key}.",
                     "references": references or []})


class PineconeRetrievalTests(unittest.TestCase):
    def build(self, **options):
        docs = Mock()
        dense = [document("a", .9, ["c"]), document("b", .8, ["c"])]
        keyword = [document("b", 20, ["c"]), document("a", 10, ["c"])]
        docs.search.side_effect = lambda **kwargs: SimpleNamespace(matches=(
            dense if kwargs["score_by"][0]["type"] == "dense_vector" else keyword))
        docs.fetch.return_value = SimpleNamespace(documents={"c": document("c")})
        settings = RagSettings(k=1, n_prime=2, **options)
        search = PineconeRetriever(settings, index=SimpleNamespace(documents=docs),
                                   manifest={"embedding_model": EMBEDDING_MODEL})
        search._embedder = Mock(encode=Mock(return_value=[[1.0] * DIMENSION]))
        search._reranker = Mock(score=Mock(return_value=[0.0, 10.0]))
        return search, docs

    def test_dense_mode_uses_cosine_field_and_one_reference_fetch(self):
        search, docs = self.build(bm25=False, rerank=False)
        hits = search.search("return")
        self.assertEqual([hit["id"] for hit in hits], ["a", "c"])
        self.assertEqual(docs.search.call_count, 1)
        self.assertEqual(docs.search.call_args.kwargs["score_by"][0]["type"], "dense_vector")
        self.assertNotIn("filter", docs.search.call_args.kwargs)
        self.assertEqual(hits[0]["text"], "Policy a.")
        self.assertEqual(docs.fetch.call_args.kwargs["ids"], ["c"])
        search._embedder.encode.assert_called_once_with(["search_query: return"])
        search._reranker.score.assert_not_called()

    def test_hybrid_combines_both_searches_without_using_the_reranker(self):
        search, docs = self.build(rerank=False)
        hits = search.search("return")
        self.assertEqual([hit["id"] for hit in hits], ["a", "c"])
        self.assertEqual({call.kwargs["score_by"][0]["type"] for call in docs.search.call_args_list},
                         {"dense_vector", "text"})
        search._reranker.score.assert_not_called()

    def test_hybrid_runs_both_remote_searches_and_reranking_can_change_the_order(self):
        search, docs = self.build()
        hits = search.search("return")
        self.assertEqual([hit["id"] for hit in hits], ["b", "c"])
        self.assertEqual({call.kwargs["score_by"][0]["type"] for call in docs.search.call_args_list},
                         {"dense_vector", "text"})
        search._reranker.score.assert_called_once()
        docs.fetch.assert_called_once()

    def test_missing_references_fail_instead_of_silently_omitting_evidence(self):
        search, docs = self.build(bm25=False, rerank=False)
        docs.fetch.return_value = SimpleNamespace(documents={})
        with self.assertRaises(ValueError):
            search.search("return")

    def test_query_recipe_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "recipe"):
            PineconeRetriever(RagSettings(embedding_model="different"), index=Mock(),
                              manifest={"embedding_model": EMBEDDING_MODEL})

    def test_reference_expansion_does_not_inflate_top_k_metrics(self):
        notebook = nbformat.read(Path(__file__).resolve().parents[1] / "src/support_ops/policy/rag/evaluation.ipynb", as_version=4)
        cell = next(cell for cell in notebook.cells if "metrics" in cell.metadata.get("tags", []))
        namespace = {}
        exec(cell.source, namespace)
        score = namespace["retrieval_metrics"]
        self.assertEqual(score(["c"], ["a"], 1)["recall_at_k"], 0)
        self.assertEqual(score(["c"], ["a", "c"], 1)["recall_at_k"], 0)
        self.assertEqual(score(["c"], ["a", "c"], 2)["mrr_at_k"], 0.5)
