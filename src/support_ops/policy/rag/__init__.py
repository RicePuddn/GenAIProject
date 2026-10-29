"""RAG policy retrieval

loader.py     policy files -> clause chunks (runs once)
retriever.py  question -> most relevant clauses (BM25 + embeddings, rerank, exceptions)
policy.py     RagPolicy: the retriever as the agent's `policy` tool
"""
from support_ops.policy.rag.policy import RagPolicy

__all__ = ["RagPolicy"]
