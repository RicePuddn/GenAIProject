"""Shared embedding and reranking models, loaded once and reused."""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DIMENSION = 768


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


# Import the model library only when a model is needed.
def _sentence_transformers():
    try:
        import sentence_transformers
    except ImportError as exc:
        raise RuntimeError("RAG search needs the rag extra: pip install -e '.[rag]'") from exc
    return sentence_transformers


class SentenceTransformerEmbedder:
    def __init__(self, name: str):
        self.name = name
        self.model = _sentence_transformers().SentenceTransformer(name)

    # Reject oversized input, then embed and normalize without truncating policy text.
    def encode(self, texts: list[str]) -> list[list[float]]:
        import torch.nn.functional as F

        for text in texts:
            if len(self.model.tokenizer.encode(text, truncation=False)) > self.model.max_seq_length:
                raise ValueError("text exceeds the embedding model input limit; revise the source section")
        vectors = self.model.encode(texts, batch_size=16, convert_to_tensor=True, show_progress_bar=False)
        if self.name == EMBEDDING_MODEL:
            vectors = F.layer_norm(vectors, normalized_shape=(vectors.shape[1],))
        return F.normalize(vectors, p=2, dim=1).cpu().tolist()


class CrossEncoderReranker:
    def __init__(self, name: str):
        self.model = _sentence_transformers().CrossEncoder(name)

    # Score each query-and-passage pair; the retriever sorts these scores.
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(score) for score in self.model.predict(
            [[query, passage] for passage in passages], batch_size=8, show_progress_bar=False,
        )]


# Cache model instances so repeated tool calls do not reload their weights.
@lru_cache(maxsize=4)
def load_embedder(name: str) -> Embedder:
    return SentenceTransformerEmbedder(name)


# Use the same reranker instance for startup warm-up and later searches.
@lru_cache(maxsize=4)
def load_reranker(name: str) -> Reranker:
    return CrossEncoderReranker(name)
