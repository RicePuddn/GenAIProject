"""Rebuild the RAG index after a policy change: chunk and embed the policy PDFs, then upload to Pinecone.

    .venv/bin/python scripts/build_rag_index.py            # embed, then upload
    .venv/bin/python scripts/build_rag_index.py --no-upload

Embedding reuses saved vectors for unchanged clauses. Uploading needs PINECONE_API_KEY and verifies the
records it uploaded.
"""
from __future__ import annotations

import argparse
import json

from support_ops.shared.config import ROOT, load_config
from support_ops.shared.policy.rag.embeddings import EMBEDDING_DIR, prepare_embeddings
from support_ops.shared.policy.rag.pinecone_store import upload_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-upload", action="store_true", help="only chunk and embed")
    args = parser.parse_args()
    manifest = prepare_embeddings(ROOT / load_config()["policy"]["corpus_dir"])
    print(f"Saved {manifest['count']} embeddings ({manifest['dimension']} dimensions) in {EMBEDDING_DIR}")
    if not args.no_upload:
        print(json.dumps(upload_embeddings(), indent=2))


if __name__ == "__main__":
    main()
