"""Offline RAG checks: saved embeddings, retrieval, evaluation metrics and agent tools."""
import tempfile
import json
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
from support_ops.policy.rag.embeddings import prepare_embeddings, read_embeddings
from support_ops.policy.rag.loader import Clause
from support_ops.policy.rag.models import DIMENSION, EMBEDDING_MODEL
from support_ops.policy.rag.pinecone_store import upload_embeddings, verify_upload
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
            build.return_value.search.return_value = [{"id": "RET-SupportingEvidence-3.2"}]
            result = retrieve("Evidence is unsafe")
            settings = build.call_args.args[0]
            self.assertEqual(settings, RagSettings.from_config(load_config()["policy"]["rag"]))
            build.return_value.search.assert_called_once_with("Evidence is unsafe")
            self.assertEqual(result[0]["id"], "RET-SupportingEvidence-3.2")

    def test_invalid_settings_fail(self):
        for values in ({"dense": False, "bm25": False}, {"k": 30}, {"typo": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RagSettings.from_config(values)


class RagToolTests(unittest.TestCase):
    def setUp(self):
        self.engine = Mock()
        self.engine.sections.return_value = ["Orders and Shipping Policy", "Returns and Refunds Policy"]
        self.engine.search.return_value = [{"id": "RET-ReturnEligibility-2.5", "text": "Faulty final-sale goods may qualify."}]
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
    def test_cache_reuses_vectors_and_invalidates_changed_policy_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clause = Clause(id="returns-policy-1", title="Returns", section="Returns Policy",
                            source="returns-policy.pdf", section_id="1", text="Returns are accepted.")
            embedder = Mock(encode=Mock(return_value=[[1.0] + [0.0] * (DIMENSION - 1)]))
            with patch("support_ops.policy.rag.embeddings.ROOT", root), patch(
                "support_ops.policy.rag.embeddings.load_corpus", return_value=[clause]
            ) as load:
                first = prepare_embeddings(root / "pdf", root / "embeddings", embedder)
                self.assertEqual(first, prepare_embeddings(root / "pdf", root / "embeddings", embedder))
                self.assertEqual(embedder.encode.call_count, 1)
                manifest, documents = read_embeddings(root / "embeddings")
                self.assertEqual(manifest["embedding_model"], EMBEDDING_MODEL)
                self.assertEqual(len(documents[0]["embedding"]), DIMENSION)
                self.assertNotIn("text", documents[0])
                self.assertTrue(embedder.encode.call_args.args[0][0].startswith("search_document: Returns Policy"))
                load.return_value = [Clause(**{**clause.__dict__, "text": "Returns require review."})]
                changed = prepare_embeddings(root / "pdf", root / "embeddings", embedder)
                self.assertNotEqual(first["corpus_hash"], changed["corpus_hash"])
                self.assertEqual(embedder.encode.call_count, 2)
                (root / "embeddings/policies.jsonl").write_text("tampered")
                with self.assertRaisesRegex(ValueError, "manifest"):
                    read_embeddings(root / "embeddings")
                self.assertEqual(changed, prepare_embeddings(root / "pdf", root / "embeddings", embedder))
                self.assertEqual(embedder.encode.call_count, 3)

                manifest_path = root / "embeddings/manifest.json"
                wrong_recipe = {**changed, "document_prefix": "wrong_prefix: "}
                manifest_path.write_text(json.dumps(wrong_recipe))
                with self.assertRaisesRegex(ValueError, "recipe"):
                    read_embeddings(root / "embeddings")
                self.assertEqual(changed, prepare_embeddings(root / "pdf", root / "embeddings", embedder))
                self.assertEqual(embedder.encode.call_count, 4)

    def test_invalid_dimensions_do_not_produce_uploadable_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clause = Clause(id="policy-1", title="Returns", section="Policy", source="policy.pdf", text="Return goods.")
            with patch("support_ops.policy.rag.embeddings.ROOT", root), patch(
                "support_ops.policy.rag.embeddings.load_corpus", return_value=[clause]
            ), self.assertRaisesRegex(ValueError, "768"):
                prepare_embeddings(root / "pdf", root / "embeddings", Mock(encode=Mock(return_value=[[1.0]])))
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
        with patch("support_ops.policy.rag.embeddings.read_embeddings", return_value=({"embedding_model": EMBEDDING_MODEL}, [])), patch(
            "support_ops.policy.rag.pinecone_store.verify_upload"
        ):
            search = PineconeRetriever(settings, index=SimpleNamespace(documents=docs))
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
        with patch("support_ops.policy.rag.embeddings.read_embeddings", return_value=({"embedding_model": EMBEDDING_MODEL}, [])):
            with self.assertRaisesRegex(ValueError, "recipe"):
                PineconeRetriever(RagSettings(embedding_model="different"), index=Mock())

    def test_startup_warm_up_loads_and_exercises_models_without_searching(self):
        search, docs = self.build()
        with patch("support_ops.policy.rag.retriever.load_embedder") as embed, patch(
            "support_ops.policy.rag.retriever.load_reranker"
        ) as rerank:
            search.warm_up()
            embed.return_value.encode.assert_called_once_with(["search_query: warm-up"])
            rerank.return_value.score.assert_called_once()
        docs.search.assert_not_called()

    def test_reference_expansion_does_not_inflate_top_k_metrics(self):
        notebook = nbformat.read(Path(__file__).resolve().parents[1] / "src/support_ops/policy/rag/evaluation.ipynb", as_version=4)
        cell = next(cell for cell in notebook.cells if "metrics" in cell.metadata.get("tags", []))
        namespace = {}
        exec(cell.source, namespace)
        score = namespace["retrieval_metrics"]
        self.assertEqual(score(["c"], ["a"], 1)["recall_at_k"], 0)
        self.assertEqual(score(["c"], ["a", "c"], 1)["recall_at_k"], 0)
        self.assertEqual(score(["c"], ["a", "c"], 2)["mrr_at_k"], 0.5)

    def test_reference_checks_accept_existing_search_hits_and_report_missing_links(self):
        notebook = nbformat.read(Path(__file__).resolve().parents[1] / "src/support_ops/policy/rag/evaluation.ipynb", as_version=4)
        cell = next(cell for cell in notebook.cells if "reference_checks" in cell.metadata.get("tags", []))
        namespace = {}
        exec(cell.source, namespace)
        check = namespace["check_references"]
        for via in ("search", "reference"):
            with self.subTest(via=via):
                self.assertTrue(check(["c"], [{"id": "c", "via": via}])["references_returned"])
        missing = check(["c"], [{"id": "a", "via": "search"}])
        self.assertFalse(missing["references_returned"])
        self.assertEqual(missing["missing_reference_chunk_ids"], ["c"])
        self.assertIsNone(check([], [])["references_returned"])


class UploadVerificationTests(unittest.TestCase):
    def test_upload_retires_old_ids_and_verifies_the_renamed_records(self):
        document = {"_id": "ODR-OrderChangeCancel-4.1", "source": "orders-and-shipping-policy.pdf",
                    "search_text": "Order changes.", "references": [], "embedding": [1.0] * DIMENSION}
        old_id = "orders-and-shipping-policy-4.1"
        live_ids = {old_id, document["_id"]}
        docs = Mock()
        docs.batch_upsert.return_value = SimpleNamespace(errors=[])
        docs.list.side_effect = lambda **args: [SimpleNamespace(id=key) for key in live_ids
                                              if key.startswith(args["prefix"])]
        docs.delete.side_effect = lambda **args: live_ids.difference_update(args["ids"])
        docs.fetch.return_value = SimpleNamespace(documents={document["_id"]: document})
        manifest = {"corpus_hash": "renamed", "data_sha256": "vectors", "count": 1, "dimension": DIMENSION}
        with tempfile.TemporaryDirectory() as directory, patch(
            "support_ops.policy.rag.pinecone_store.read_embeddings", return_value=(manifest, [document])
        ), patch("support_ops.policy.rag.pinecone_store.pinecone_client") as client:
            client.return_value.index.return_value = SimpleNamespace(documents=docs)
            receipt = upload_embeddings(Path(directory))
        self.assertEqual(receipt["count"], 1)
        self.assertEqual(live_ids, {document["_id"]})
        docs.delete.assert_called_once_with(namespace="policies", ids=[old_id])

    def test_manifest_receipt_and_live_records_must_all_agree(self):
        document = {"_id": "policy-1", "source": "policy.pdf", "search_text": "Original policy.",
                    "references": [], "embedding": [0.5, 0.5]}
        manifest = {"corpus_hash": "corpus-a", "data_sha256": "vectors-a", "count": 1, "dimension": 2}
        receipt = {**manifest, "index": "support-policies", "namespace": "policies"}
        docs = Mock()
        actual = dict(document)
        docs.fetch.return_value = SimpleNamespace(documents={"policy-1": actual})
        docs.list.return_value = [SimpleNamespace(id="policy-1")]
        index = SimpleNamespace(documents=docs)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "pinecone_upload.json"
            path.write_text(json.dumps(receipt))
            verify_upload(index, "policies", manifest, [document], folder)
            path.write_text(json.dumps({**receipt, "corpus_hash": "old-corpus"}))
            with self.assertRaisesRegex(ValueError, "manifest"):
                verify_upload(index, "policies", manifest, [document], folder)
            path.write_text(json.dumps(receipt))
            actual["search_text"] = "Stale remote policy."
            with self.assertRaisesRegex(ValueError, "Pinecone records"):
                verify_upload(index, "policies", manifest, [document], folder)
            actual["search_text"] = document["search_text"]
            actual["embedding"] = [0.1, 0.9]
            with self.assertRaisesRegex(ValueError, "Pinecone records"):
                verify_upload(index, "policies", manifest, [document], folder)
