"""Create reusable, upload-ready policy embeddings. No Pinecone writes happen here."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from support_ops.helpers.config import ROOT
from support_ops.policy.rag.loader import load_corpus, write_chunks
from support_ops.policy.rag.models import DIMENSION, EMBEDDING_MODEL, Embedder, load_embedder

EMBEDDING_DIR = ROOT / "policy/embed_chunk"
EMBEDDING_RECIPE = {
    "embedding_model": EMBEDDING_MODEL,
    "dimension": DIMENSION,
    "metric": "cosine",
    "contextual": True,
    "document_prefix": "search_document: ",
    "query_prefix": "search_query: ",
    "normalization": "layer_norm_then_l2",
}


# Fingerprint the exact chunk records so changed policies cannot reuse stale embeddings.
def corpus_fingerprint(records: list[dict]) -> str:
    content = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


# Regenerate chunks, embed their contextual text, and save every vector alongside its source.
def prepare_embeddings(source: Path, output: Path = EMBEDDING_DIR,
                       embedder: Embedder | None = None) -> dict:
    # Step 1: Create the current section chunks and their resolved reference IDs.
    clauses = load_corpus(source)
    paths = write_chunks(clauses, ROOT / "policy/chunk_policy")
    records = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    corpus_hash = corpus_fingerprint(records)
    output.mkdir(parents=True, exist_ok=True)

    # Step 2: Reuse saved vectors only when the complete corpus and model recipe match.
    try:
        manifest, _ = read_embeddings(output)
    except (FileNotFoundError, ValueError):
        manifest = None  # Missing or incompatible cached data must be rebuilt.
    if manifest is not None and manifest["corpus_hash"] == corpus_hash:
        return manifest

    # Step 3: Use the same model and preprocessing later used for query embeddings.
    texts = [clause.indexed_text() for clause in clauses]
    vectors = (embedder or load_embedder(EMBEDDING_MODEL)).encode([f"search_document: {text}" for text in texts])
    if len(vectors) != len(records):
        raise ValueError("embedding count does not match the chunk count")
    documents = []
    for record, search_text, vector in zip(records, texts, vectors):
        if len(vector) != DIMENSION or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"invalid 768-dimensional embedding for {record['id']}")
        document = {"_id": record["id"],
                    **{key: value for key, value in record.items() if key not in {"id", "text"}},
                    "search_text": search_text, "embedding": vector}
        documents.append(document)

    # Step 4: Save upload-ready JSONL and a manifest describing the embedding recipe.
    data = "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents)
    (output / "policies.jsonl").write_text(data, encoding="utf-8")
    manifest = {"corpus_hash": corpus_hash, **EMBEDDING_RECIPE, "count": len(documents),
                "data_sha256": hashlib.sha256(data.encode()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# Validate the saved model recipe, file integrity, IDs and references before upload or search.
def read_embeddings(folder: Path = EMBEDDING_DIR) -> tuple[dict, list[dict]]:
    manifest = json.loads((folder / "manifest.json").read_text())
    data = (folder / "policies.jsonl").read_text(encoding="utf-8")
    if any(manifest.get(key) != value for key, value in EMBEDDING_RECIPE.items()):
        raise ValueError("saved embeddings use a different model recipe; regenerate them")
    if hashlib.sha256(data.encode()).hexdigest() != manifest["data_sha256"]:
        raise ValueError("saved embedding file does not match its manifest")
    documents = [json.loads(line) for line in data.splitlines() if line.strip()]
    ids = {doc["_id"] for doc in documents}
    if not documents or len(ids) != len(documents) or len(documents) != manifest["count"]:
        raise ValueError("invalid or duplicate saved embedding records")
    for doc in documents:
        if len(doc["embedding"]) != DIMENSION or not all(math.isfinite(v) for v in doc["embedding"]):
            raise ValueError(f"invalid embedding for {doc['_id']}")
        if set(doc["references"]) - ids:
            raise ValueError(f"unresolved references in {doc['_id']}")
    return manifest, documents


if __name__ == "__main__":
    from support_ops.helpers.config import load_config

    policy = load_config()["policy"]
    manifest = prepare_embeddings(ROOT / policy["corpus_dir"])
    print(f"Saved {manifest['count']} embeddings ({manifest['dimension']} dimensions) in {EMBEDDING_DIR}")
