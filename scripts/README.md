# Scripts

- `seed_db.py` holds every seed row, grouped by table, and inserts them after creating the tables from
  `data/db/schema.sql`. Existing rows are kept; `--reset` deletes all rows first, agent writes included.
  See the [fixture README](../data/db/README.md) for sample IDs.
- `sync_skills.py` refreshes the skills clause files in `data/policy/skills/` from the policy PDFs, so
  RAG and skills use the same clauses. Run it after reviewing a policy change.
- `build_rag_index.py` chunks and embeds the policy PDFs (reusing saved vectors for unchanged clauses),
  then uploads them to Pinecone and verifies the upload. `--no-upload` only embeds.

Run each from the repository root after configuring `.env`, e.g. `.venv/bin/python scripts/seed_db.py`.
