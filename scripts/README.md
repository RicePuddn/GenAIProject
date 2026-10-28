# Scripts

`seed_backend.py` creates missing PostgreSQL tables and inserts deterministic
fictional ecommerce fixtures. Run it from the repository root after configuring
`.env` and starting the database:

```bash
.venv/bin/python scripts/seed_backend.py
```

See the [application README](../README.md) for setup and the
[fixture README](../data/backend_seed/README.md) for sample IDs and reset instructions.
