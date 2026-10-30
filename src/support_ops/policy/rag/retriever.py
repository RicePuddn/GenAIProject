"""Shared policy retrieval: cosine, hybrid RRF, or hybrid with BGE reranking.

Three approaches:
1. vector: vector search using cosine similarity.
2. hybrid: BM25 + cosine vector search, combined using RRF.
3. hybrid_rerank: the same hybrid search, followed by BGE reranking.

Agents/tools call retrieve(query) using the organisation configuration.
Evaluation compares all three approaches through the same PineconeRetriever.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any

from support_ops.policy.rag.models import EMBEDDING_MODEL, RERANK_MODEL, load_embedder, load_reranker

# search_text is the existing index's BM25 field and the only stored copy of chunk text.
FIELDS = ["source", "domain", "category", "section_id", "section_title", "search_text", "references"]
APPROACHES = {
    # Approach 1: Vector search using cosine similarity.
    "vector": {"dense": True, "bm25": False, "rerank": False},
    # Approach 2: BM25 + vector search using cosine similarity, combined with RRF.
    "hybrid": {"dense": True, "bm25": True, "rerank": False},
    # Approach 3: The same hybrid search + BGE reranking.
    "hybrid_rerank": {"dense": True, "bm25": True, "rerank": True},
}


@dataclass(frozen=True)
class RagSettings:
    dense: bool = True               # embedding search
    bm25: bool = True                # keyword search; with dense, fused by RRF
    rrf_k: int = 60
    n_prime: int = 20                # candidates kept for the reranker
    rerank: bool = True              # cross-encoder over the N' candidates
    contextual: bool = True          # prepend the section name to each chunk before indexing
    k: int = 4                       # ranked clauses, before adding direct references
    expand_references: bool = True   # include directly referenced sections after top K
    embedding_model: str = EMBEDDING_MODEL
    rerank_model: str = RERANK_MODEL

    def __post_init__(self) -> None:
        flags = {"dense": self.dense, "bm25": self.bm25, "rerank": self.rerank}
        if flags not in APPROACHES.values():
            raise ValueError("policy.rag must select vector, hybrid, or hybrid_rerank settings")
        if not 1 <= self.k <= self.n_prime:
            raise ValueError("policy.rag needs 1 <= k <= n_prime")

    @classmethod
    def from_config(cls, rag: dict[str, Any]) -> RagSettings:
        unknown = set(rag) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown policy.rag settings: {sorted(unknown)}")
        return cls(**rag)


# ---------------------------------------------------------------- scoring

def ranked(scores: dict[str, float]) -> list[str]:
    """Chunk IDs by descending score, with reproducible ties."""
    return sorted(scores, key=lambda index: (-scores[index], index))


def reciprocal_rank_fusion(rankings: list[list[str]], k: int) -> dict[str, float]:
    """RRF: each list adds 1 / (k + rank). Uses ranks only, so BM25 scores and
    cosine similarities never need to be put on the same scale."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            fused[index] += 1 / (k + rank)
    return fused


# ---------------------------------------------------------------- retrieval

# Convert Pinecone documents into the existing agent policy-tool result format.
def document_hit(document, score=None, via: str = "search") -> dict:
    hit = {field: document.get(field) for field in FIELDS if field != "search_text"}
    hit.update(
        id=document.id,
        title=document.get("section_title"),
        section=document.get("domain"),
        score=score,
        via=via,
        text=document.get("search_text"),
        references=document.get("references", []),
    )
    return hit


class PineconeRetriever:
    # Connect to an existing index. This class never creates an index or uploads data.
    def __init__(self, settings: RagSettings, *, index=None):
        self.settings = settings
        from support_ops.policy.rag.embeddings import read_embeddings
        from support_ops.policy.rag.pinecone_store import pinecone_client, verify_upload

        self.manifest, documents = read_embeddings()
        if settings.embedding_model != self.manifest["embedding_model"] or not settings.contextual:
            raise ValueError("query model/context settings must match the saved document embedding recipe")
        self.index = index if index is not None else pinecone_client().index(
            os.environ.get("PINECONE_INDEX", "support-policies"))
        self.namespace = os.environ.get("PINECONE_NAMESPACE", "policies")
        verify_upload(self.index, self.namespace, self.manifest, documents)
        self._embedder, self._reranker = None, None

    # Give the planner only the domains available in this fixed policy corpus.
    def sections(self) -> list[str]:
        return ["Orders and Shipping Policy", "Returns and Refunds Policy"]

    # Load and exercise the selected models before the server starts timing tickets.
    def warm_up(self) -> None:
        self._embedder = load_embedder(self.settings.embedding_model)
        self._embedder.encode(["search_query: warm-up"])
        if self.settings.rerank:
            self._reranker = load_reranker(self.settings.rerank_model)
            self._reranker.score("warm-up", ["Policy retrieval warm-up."])

    # Ask the existing index for cosine matches or BM25 keyword matches.
    def _search(self, scoring: dict) -> list[dict]:
        response = self.index.documents.search(
            namespace=self.namespace, top_k=self.settings.n_prime, score_by=[scoring],
            include_fields=FIELDS,
        )
        return [document_hit(doc, float(doc.score)) for doc in response.matches]

    # Approach 1: Embed the query; Pinecone ranks matches using the index's cosine metric.
    def _vector_search(self, query: str) -> list[dict]:
        self._embedder = self._embedder or load_embedder(self.settings.embedding_model)
        vector = self._embedder.encode([f"search_query: {query}"])[0]
        hits = self._search({"type": "dense_vector", "field": "embedding", "values": vector})
        return sorted(hits, key=lambda hit: (-hit["score"], hit["id"]))

    # Get BM25 keyword matches from Pinecone for the hybrid approaches.
    def _bm25_search(self, query: str) -> list[dict]:
        return self._search({"type": "text", "fields": ["search_text"], "query": query})

    # Approach 2: Run cosine and BM25 searches, then combine their rankings using RRF.
    def _hybrid_search(self, query: str) -> list[dict]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            vector = pool.submit(self._vector_search, query)
            keyword = pool.submit(self._bm25_search, query)
            lists = [vector.result(), keyword.result()]
        documents = {hit["id"]: hit for matches in lists for hit in matches}
        scores = reciprocal_rank_fusion(
            [[hit["id"] for hit in matches] for matches in lists], self.settings.rrf_k,
        )
        return [{**documents[key], "score": float(scores[key])}
                for key in ranked(scores)[:self.settings.n_prime]]

    # Approach 3: Take the hybrid candidates and rank them again with the BGE reranker.
    def _hybrid_rerank_search(self, query: str) -> list[dict]:
        candidates = self._hybrid_search(query)
        if not candidates:
            return []
        self._reranker = self._reranker or load_reranker(self.settings.rerank_model)
        values = self._reranker.score(query, [hit["text"] for hit in candidates])
        scores = dict(zip([hit["id"] for hit in candidates], values))
        documents = {hit["id"]: hit for hit in candidates}
        return [{**documents[key], "score": float(scores[key])} for key in ranked(scores)]

    # Select one of the three approaches, keep the top K, then include direct references.
    def search(self, query: str) -> list[dict]:
        if not query.strip():
            raise ValueError("policy query must not be empty")

        # Step 1: Run the selected retrieval approach.
        if self.settings.rerank:
            candidates = self._hybrid_rerank_search(query)  # Approach 3
        elif self.settings.bm25:
            candidates = self._hybrid_search(query)         # Approach 2
        else:
            candidates = self._vector_search(query)         # Approach 1

        # Step 2: Keep the final top K ranked search results.
        hits = candidates[:self.settings.k]

        # Step 3: Fetch missing direct references in one batch; do not recursively expand them.
        if self.settings.expand_references:
            returned = {hit["id"] for hit in hits}
            references = dict.fromkeys(ref for hit in hits for ref in hit["references"] if ref not in returned)
            if references:
                response = self.index.documents.fetch(namespace=self.namespace, ids=list(references), include_fields=FIELDS)
                if set(references) - set(response.documents):
                    raise ValueError("Pinecone is missing referenced policy sections; upload the complete snapshot")
                for key in references:
                    doc = response.documents[key]
                    hits.append(document_hit(doc, via="reference"))
        return hits


# Reuse the configured retriever between tool calls; no repeated model or client setup.
@lru_cache(maxsize=8)
def shared_retriever(settings: RagSettings) -> PineconeRetriever:
    return PineconeRetriever(settings)


# Public entry point: use the organisation's selected retrieval settings from base.yaml.
def retrieve(query: str) -> list[dict]:
    from support_ops.helpers.config import load_config

    settings = RagSettings.from_config(load_config()["policy"]["rag"])
    return shared_retriever(settings).search(query)
