# Labelled cases

- `agents/gold.json`: 10 standalone customer requests, with expected outcomes and architecture-specific tool expectations. See [evaluation guidance](agents/README.md).
- `rag/questions.json`: retrieval-only questions labelled with relevant policy chunk IDs.

The agent cases reference the PostgreSQL rows inserted by `scripts/seed_backend.py`.
