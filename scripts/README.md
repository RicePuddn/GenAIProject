# Scripts

`seed_backend.py` creates missing PostgreSQL tables and inserts deterministic
fictional ecommerce fixtures. Run it from the repository root after configuring
`.env` and starting the database:

```bash
.venv/bin/python scripts/seed_backend.py
```

`demo_app.py` is a Streamlit chat page for the running API that shows each agent
step; see "Demo page" in the [application README](../README.md).

See the [application README](../README.md) for setup and the
[fixture README](../data/backend_seed/README.md) for sample IDs and reset instructions.

`sync_policy_skills.py` refreshes the skills clause files from the policy PDFs.
Run it after reviewing a policy change so RAG and skills use the same clauses.
