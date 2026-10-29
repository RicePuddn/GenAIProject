"""Policy RAG pipeline.

loader.py          read policy files and save section chunks
embeddings.py      embed chunks and save upload-ready records
pinecone_store.py  upload records to the existing index
retriever.py       shared retrieval for agents and evaluation
policy.py          adapter for the existing agent policy tool
evaluation.ipynb   compare retrieved chunk IDs against labelled questions
"""
from support_ops.policy.rag.policy import RagPolicy
from support_ops.policy.rag.retriever import retrieve

__all__ = ["RagPolicy", "retrieve"]
