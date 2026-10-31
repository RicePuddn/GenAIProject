"""Upload saved embeddings to an existing Pinecone index."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from support_ops.helpers.config import load_env
from support_ops.policy.rag.identifiers import POLICY_SECTION_PREFIXES
from support_ops.policy.rag.embeddings import EMBEDDING_DIR, read_embeddings
from support_ops.policy.rag.models import DIMENSION

# Read index settings from .env; never print or save the API key.
def pinecone_client():
    from pinecone import Pinecone

    load_env()
    if not os.environ.get("PINECONE_API_KEY"):
        raise ValueError("set PINECONE_API_KEY in .env")
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


# List this corpus's IDs, including old document-name IDs that must be retired.
def _stored_policy_ids(index, namespace: str, documents: list[dict]) -> set[str]:
    prefixes = set()
    for source in {doc["source"] for doc in documents}:
        prefixes.add(Path(source).stem + "-")
        prefixes.update(prefix + "-" for prefix in POLICY_SECTION_PREFIXES.get(source, {}).values())
    return {doc.id for prefix in sorted(prefixes)
            for doc in index.documents.list(namespace=namespace, prefix=prefix)}


# Compare the live records with the saved snapshot, including vector values.
def snapshot_matches(index, namespace: str, documents: list[dict]) -> bool:
    ids = [doc["_id"] for doc in documents]
    fields = [field for field in documents[0] if field != "_id"]
    fetched = index.documents.fetch(namespace=namespace, ids=ids, include_fields=fields)
    if set(fetched.documents) != set(ids):
        return False
    for expected in documents:
        actual = fetched.documents[expected["_id"]]
        if any(actual.get(field) != expected[field] for field in fields if field != "embedding"):
            return False
        vector = actual.get("embedding") or []
        if len(vector) != len(expected["embedding"]) or any(
            not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7) for a, b in zip(vector, expected["embedding"])
        ):
            return False
    return _stored_policy_ids(index, namespace, documents) == set(ids)


# Verify the receipt fingerprint and the actual namespace before serving searches.
def verify_upload(index, namespace: str, manifest: dict, documents: list[dict], folder: Path = EMBEDDING_DIR) -> None:
    receipt_path = folder / "pinecone_upload.json"
    if not receipt_path.exists():
        raise ValueError("upload receipt is missing; upload the saved embeddings first")
    receipt = json.loads(receipt_path.read_text())
    expected = {"index": os.environ.get("PINECONE_INDEX", "support-policies"), "namespace": namespace,
                "corpus_hash": manifest["corpus_hash"], "data_sha256": manifest["data_sha256"],
                "count": manifest["count"], "dimension": manifest["dimension"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("upload receipt does not match the local embedding manifest; upload again")
    if not snapshot_matches(index, namespace, documents):
        raise ValueError("Pinecone records do not match the saved corpus; upload again")


# Upload the saved snapshot, remove retired sections, then confirm the complete live corpus.
def upload_embeddings(folder: Path = EMBEDDING_DIR) -> dict:
    manifest, documents = read_embeddings(folder)
    index = pinecone_client().index(os.environ.get("PINECONE_INDEX", "support-policies"))
    namespace = os.environ.get("PINECONE_NAMESPACE", "policies")
    result = index.documents.batch_upsert(namespace=namespace, documents=documents, show_progress=False)
    if result.errors:
        raise RuntimeError(f"Pinecone rejected {len(result.errors)} batches; no upload success recorded")
    ids = {doc["_id"] for doc in documents}
    existing = _stored_policy_ids(index, namespace, documents)
    if stale := existing - ids:
        index.documents.delete(namespace=namespace, ids=sorted(stale))
    deadline = time.monotonic() + 120
    while not snapshot_matches(index, namespace, documents):
        if time.monotonic() >= deadline:
            raise TimeoutError("upload accepted, but the current corpus is not fully readable yet")
        time.sleep(2)
    receipt = {"index": os.environ.get("PINECONE_INDEX", "support-policies"), "namespace": namespace,
               "corpus_hash": manifest["corpus_hash"], "data_sha256": manifest["data_sha256"],
               "count": len(ids), "dimension": DIMENSION}
    (folder / "pinecone_upload.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    print(json.dumps(upload_embeddings(), indent=2))
