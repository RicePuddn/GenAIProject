"""Score clauses against a question (BM25 + embeddings, fused), rerank the
best few, and add the exceptions of whatever is returned.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from support_ops.policy.rag.loader import Clause, load_corpus

TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class RagSettings:
    chunking: str = "clause"       
    window_words: int = 180
    window_overlap: int = 30
    dense: bool = True               # embedding search
    bm25: bool = True                # keyword search; with dense, fused by RRF
    rrf_k: int = 60
    n_prime: int = 20                # candidates kept for the reranker
    rerank: bool = True              # cross-encoder over the N' candidates
    contextual: bool = True          # prepend the section name to each chunk before indexing
    k: int = 4                       # clauses returned to the agent
    expand_exceptions: bool = True   # also return exceptions of the returned clauses
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __post_init__(self) -> None:
        if self.chunking not in {"clause", "window"}:
            raise ValueError("policy.rag.chunking must be clause or window")
        if not (self.dense or self.bm25):
            raise ValueError("policy.rag needs dense or bm25 search")
        if not 1 <= self.k <= self.n_prime:
            raise ValueError("policy.rag needs 1 <= k <= n_prime")
        if not 0 <= self.window_overlap < self.window_words:
            raise ValueError("policy.rag needs 0 <= window_overlap < window_words")

    @classmethod
    def from_config(cls, rag: dict[str, Any]) -> RagSettings:
        unknown = set(rag) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown policy.rag settings: {sorted(unknown)}")
        return cls(**rag)


# ---------------------------------------------------------------- scoring

def tokenize(text: str) -> list[str]:
    """Lower-case words; hyphenated terms such as RET-1 or final-sale count whole and in parts."""
    tokens = []
    for token in TOKEN.findall(text.lower()):
        tokens.append(token)
        if "-" in token:
            tokens += token.split("-")
    return tokens


class BM25:
    """Okapi BM25: TF-IDF with saturating term frequency (k1) and
    document-length normalisation (b)."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.counts = [Counter(tokenize(document)) for document in documents]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = sum(self.lengths) / len(self.lengths) or 1.0
        frequency = Counter(term for count in self.counts for term in count)
        self.idf = {term: math.log(1 + (len(documents) - n + 0.5) / (n + 0.5)) for term, n in frequency.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        return [
            sum(self.idf[term] * count[term] * (self.k1 + 1)
                / (count[term] + self.k1 * (1 - self.b + self.b * length / self.average))
                for term in terms if term in count)
            for count, length in zip(self.counts, self.lengths)
        ]


def cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def ranked(scores: dict[int, float]) -> list[int]:
    """Indices by descending score; ties keep corpus order so results are reproducible."""
    return sorted(scores, key=lambda index: (-scores[index], index))


def reciprocal_rank_fusion(rankings: list[list[int]], k: int) -> dict[int, float]:
    """RRF: each list adds 1 / (k + rank). Uses ranks only, so BM25 scores and
    cosine similarities never need to be put on the same scale."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            fused[index] += 1 / (k + rank)
    return fused


# ---------------------------------------------------------------- models

class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


def _sentence_transformers():
    try:
        import sentence_transformers
    except ImportError as exc:
        raise RuntimeError("RAG search needs the rag extra: pip install -e '.[rag]'") from exc
    return sentence_transformers


class SentenceTransformerEmbedder:
    def __init__(self, name: str):
        self.model = _sentence_transformers().SentenceTransformer(name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()


class CrossEncoderReranker:
    def __init__(self, name: str):
        self.model = _sentence_transformers().CrossEncoder(name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(score) for score in self.model.predict([[query, passage] for passage in passages])]


@lru_cache(maxsize=4)
def load_embedder(name: str) -> Embedder:
    return SentenceTransformerEmbedder(name)


@lru_cache(maxsize=4)
def load_reranker(name: str) -> Reranker:
    return CrossEncoderReranker(name)


# ---------------------------------------------------------------- retrieval

class PolicyRetriever:
    def __init__(self, clauses: list[Clause], settings: RagSettings,
                 embedder: Embedder | None = None, reranker: Reranker | None = None):
        self.clauses, self.settings = clauses, settings
        self.texts = [clause.indexed_text(settings.contextual) for clause in clauses]
        self.bm25 = BM25(self.texts) if settings.bm25 else None
        self._embedder, self._reranker, self._vectors = embedder, reranker, None
        self.exceptions: dict[str, list[int]] = defaultdict(list)
        for index, clause in enumerate(clauses):
            for parent in clause.exception_of:
                self.exceptions[parent].append(index)

    def sections(self) -> list[str]:
        return list(dict.fromkeys(clause.section for clause in self.clauses))

    def _dense_scores(self, query: str) -> dict[int, float]:
        if self._embedder is None:
            self._embedder = load_embedder(self.settings.embedding_model)
        if self._vectors is None:
            self._vectors = self._embedder.encode(self.texts)   # offline phase, done once
        query_vector = self._embedder.encode([query])[0]
        return {index: cosine(query_vector, vector) for index, vector in enumerate(self._vectors)}

    def _rerank_scores(self, query: str, candidates: list[int]) -> dict[int, float]:
        if self._reranker is None:
            self._reranker = load_reranker(self.settings.rerank_model)
        return dict(zip(candidates, self._reranker.score(query, [self.texts[i] for i in candidates])))

    def warm_up(self) -> None:
        """Load the models and embed the corpus now, so the first search is as fast as the rest
        and model loading never counts towards a ticket's latency."""
        if self.settings.dense:
            self._dense_scores("warm-up")
        if self.settings.rerank and self.texts:
            self._rerank_scores("warm-up", [0])

    def search(self, query: str) -> list[dict[str, Any]]:
        settings = self.settings
        stages: list[dict[int, float]] = []
        if settings.dense:
            stages.append(self._dense_scores(query))
        if settings.bm25:
            # Clauses sharing no term with the query are not keyword matches, so they get no rank.
            stages.append({index: score for index, score in enumerate(self.bm25.scores(query)) if score > 0})
        scores = (reciprocal_rank_fusion([ranked(stage) for stage in stages], settings.rrf_k)
                  if len(stages) == 2 else stages[0])
        candidates = ranked(scores)[:settings.n_prime]
        if settings.rerank and candidates:
            scores = self._rerank_scores(query, candidates)
            candidates = ranked(scores)
        top = candidates[:settings.k]
        hits = [self._hit(index, scores[index], "search") for index in top]
        if settings.expand_exceptions:
            returned = set(top)
            for index in top:
                for exception in self.exceptions[self.clauses[index].id]:
                    if exception not in returned:
                        returned.add(exception)
                        hits.append(self._hit(exception, None, f"exception_of:{self.clauses[index].id}"))
        return hits

    def _hit(self, index: int, score: float | None, via: str) -> dict[str, Any]:
        clause = self.clauses[index]
        return {
            "id": clause.id, "title": clause.title, "section": clause.section, "source": clause.source,
            "text": clause.text, "score": None if score is None else round(score, 4), "via": via,
        }


@lru_cache(maxsize=8)
def shared_retriever(folder: str, settings: RagSettings) -> PolicyRetriever:
    """One retriever per corpus and settings per process, so the API reuses the embedded index."""
    clauses = load_corpus(Path(folder), settings.chunking, settings.window_words, settings.window_overlap)
    return PolicyRetriever(clauses, settings)
