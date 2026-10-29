"""Upload saved embeddings to an existing Pinecone index."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from support_ops.helpers.config import load_env
from support_ops.policy.rag.embeddings import DIMENSION, EMBEDDING_DIR, read_embeddings

# Read index settings from .env; never print or save the API key.
def pinecone_client():
    from pinecone import Pinecone

    load_env()
    if not os.environ.get("PINECONE_API_KEY"):
        raise ValueError("set PINECONE_API_KEY in .env")
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


# Upload the saved snapshot and wait until every expected record is readable.
def upload_embeddings(folder: Path = EMBEDDING_DIR) -> dict:
    manifest, documents = read_embeddings(folder)
    index = pinecone_client().index(os.environ.get("PINECONE_INDEX", "support-policies"))
    namespace = os.environ.get("PINECONE_NAMESPACE", "policies")
    result = index.documents.batch_upsert(namespace=namespace, documents=documents, show_progress=False)
    if result.errors:
        raise RuntimeError(f"Pinecone rejected {len(result.errors)} batches; no upload success recorded")
    ids = [doc["_id"] for doc in documents]
    expected = {doc["_id"]: doc["search_text"] for doc in documents}
    deadline = time.monotonic() + 120
    while True:
        fetched = index.documents.fetch(namespace=namespace, ids=ids, include_fields=["search_text"])
        if len(fetched.documents) == len(ids) and all(
            doc.get("search_text") == expected[key] for key, doc in fetched.documents.items()
        ):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("upload accepted, but the current corpus is not fully readable yet")
        time.sleep(2)
    # Remove retired sections from these policy documents only, keeping this namespace current.
    prefixes = {Path(doc["source"]).stem + "-" for doc in documents}
    existing = {doc.id for prefix in prefixes for doc in index.documents.list(namespace=namespace, prefix=prefix)}
    if stale := existing - set(ids):
        index.documents.delete(namespace=namespace, ids=sorted(stale))
    receipt = {"index": os.environ.get("PINECONE_INDEX", "support-policies"), "namespace": namespace,
               "corpus_hash": manifest["corpus_hash"], "count": len(ids), "dimension": DIMENSION}
    (folder / "pinecone_upload.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    print(json.dumps(upload_embeddings(), indent=2))
