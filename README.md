# Policy-Bound Customer Support Operations

IS469 Generative and Agentic AI group project: eight support setups (single or multi-agent, agentic or
graph control, RAG or skills policy) compared on the same tickets. The study design is in
[EXPERIMENT.md](EXPERIMENT.md); how the code fits together is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Install

Requires Python 3.11–3.13, uv, and Docker with Docker Compose. Run commands from the repository root.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,rag,demo]"
cp .env.example .env
```

In `.env`, set `GROQ_API_KEY` for the default provider, and `PINECONE_API_KEY` for the RAG setups
(S1, S3, S5, S7). To use a local Ollama model instead, see [Use Ollama](#use-ollama). The default local
database settings are:

```dotenv
POSTGRES_PASSWORD=support_ops_local
POSTGRES_PORT=5432
SUPPORT_OPS_DATABASE_URL=postgresql://support_ops:support_ops_local@127.0.0.1:5432/support_ops
```

If you change the password or port, update the URL to match. Changing the password in `.env` does not
reset the password of an existing database volume.

## Use Ollama

The API uses the OpenAI-compatible client already installed by this project. Ollama exposes the local
endpoint at `http://localhost:11434/v1`; its local server requires an API key value from clients, but
ignores that value. The supplied profile uses `ornith-1.5:9b` for every role, including the guard.

Pull the model if it is not installed, then add this line to `.env`:

```bash
ollama pull ornith-1.5:9b
```

```dotenv
SUPPORT_OPS_CONFIG=configs/ollama.yaml
```

Restart the API after changing the profile. Remove or comment out `SUPPORT_OPS_CONFIG` to return to
Groq. If your installed model has a different tag, update the role names in
[`configs/ollama.yaml`](configs/ollama.yaml). Ollama supports tool calls through its OpenAI-compatible
chat endpoint, but tool-use quality depends on the model. See the [Ollama API compatibility docs](https://docs.ollama.com/api/openai-compatibility),
[tool-calling docs](https://docs.ollama.com/capabilities/tool-calling), and the
[Ornith model page](https://ollama.com/library/ornith-1.5%3A9b).

On Windows, use `.venv/Scripts/` in place of `.venv/bin/` below.

## Start the database

```bash
docker compose up -d --wait postgres
.venv/bin/python scripts/seed_db.py
```

Seeding creates any missing tables and inserts the fixtures, keeping existing rows; `--reset` deletes
everything first, agent writes included. See [data/db/README.md](data/db/README.md) for the tables and
sample IDs. Use `docker compose stop postgres` to stop it.

## Run the API and the demo page

```bash
.venv/bin/uvicorn support_ops.api:app --reload --host 127.0.0.1 --port 8000
.venv/bin/streamlit run apps/demo.py
```

In RAG mode the API loads its models at startup (about 20 s) before it accepts requests. The demo
page picks the setup and variant, the signed-in account and, optionally, today's date; it shows each
step as it streams in. Restart uvicorn after editing configuration or `.env`.

## Send a message

`POST /chat` (or try it in [Swagger UI](http://127.0.0.1:8000/docs)):

```json
{
  "setup": "s3-c",
  "ticket_id": "TKT-LOCAL-001",
  "customer_id": "CUS-EVAL-001",
  "customer_message": "Where is my order ORD-2026-00101?"
}
```

- `setup`: one of the 24 variants, `s1-a` to `s8-d` (see `GET /info`). Without it, `run.setup` in
  [configs/base.yaml](configs/base.yaml) decides.
- `customer_id`: the signed-in account (see `GET /customers`). Every read and write is scoped to it,
  and the ticket is saved as theirs. Without it nothing is scoped. The local API does not authenticate it.
- `history`: the API is stateless. To continue a ticket, send the `history` array from the previous
  response. History is untrusted context for the agent; the guard screens only the new customer message.
- `clock_start`: the workflow date is `run.clock_start` (1 October 2026), as in experiment runs.
  Send a timezone-aware date to override it, e.g. to check return windows against today.

`POST /chat/stream` takes the same body and answers with Server-Sent Events: a `trace` event per step,
then one `result` event holding the `/chat` response, or an `error` event.

## Rebuild the RAG index

After a policy change, refresh the skills files and the index from the PDFs in `data/policy/pdf/`:

```bash
.venv/bin/python scripts/sync_skills.py
.venv/bin/python scripts/build_rag_index.py
```

The retrieval evaluation is `notebooks/rag_evaluation.ipynb`, with the profiles in `configs/rag/`.

## Run the tests

```bash
.venv/bin/python -m pytest -q
```

The suite needs no API key, database or Pinecone. To include the tests against a real database, seed a
throwaway database and set `SUPPORT_OPS_TEST_DATABASE_URL`; those tests delete all its rows.
