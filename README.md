# Policy-Bound Customer Support Operations

IS469 Generative and Agentic AI group project: eight support setups (single or multi-agent, agentic or
graph control, RAG or skills policy) compared on the same tickets. The study design is in
[EXPERIMENT.md](EXPERIMENT.md); how the code fits together is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Install

Requires Python 3.11–3.13, uv, and Docker with Docker Compose. Run commands from the repository root.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[rag,demo]"
cp .env.example .env
```

The LLM provider is local Ollama; no provider API key is needed. Set `PINECONE_API_KEY` for the RAG
setups (S1, S3, S5, S7). The default local database settings are:

```dotenv
POSTGRES_PASSWORD=support_ops_local
POSTGRES_PORT=5432
SUPPORT_OPS_DATABASE_URL=postgresql://support_ops:support_ops_local@127.0.0.1:5432/support_ops
```

If you change the password or port, update the URL to match. Changing the password in `.env` does not
reset the password of an existing database volume.

## Start Ollama

The API uses the OpenAI-compatible client already installed by this project. Ollama exposes the local
endpoint at `http://localhost:11434/v1`; the client supplies a placeholder API key which the local
server ignores. The default profile uses `lfm3:latest` for every role. The optional
[`configs/ollama.yaml`](configs/ollama.yaml) profile uses `ornith:9b` for every role.

Pull the model if it is not installed:

```bash
ollama pull lfm3:latest
```

Restart Ollama and the API after changing models. If your installed model has a different tag, update the role names in
[`configs/base.yaml`](configs/base.yaml) or [`configs/ollama.yaml`](configs/ollama.yaml). Ollama supports tool calls through its OpenAI-compatible
chat endpoint, but tool-use quality depends on the model. See the [Ollama API compatibility docs](https://docs.ollama.com/api/openai-compatibility),
[tool-calling docs](https://docs.ollama.com/capabilities/tool-calling), and the
[LFM model page](https://ollama.com/library/lfm3).

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
.venv/bin/uvicorn SupportOpsBot.chatRoute:app --reload --host 127.0.0.1 --port 8000
.venv/bin/streamlit run SupportOpsBot/demo.py
```

In RAG mode the API loads its models at startup (about 20 s) before it accepts requests. The demo
page picks the setup and variant, the signed-in account and, optionally, today's date; it shows each
step as it streams in. Each completed run saves its request, response, trajectory, and an operational
summary under `results/runs/`. Restart uvicorn after editing configuration or `.env`.

## Send a message

`POST /chat` (or try it in [Swagger UI](http://127.0.0.1:8000/docs)):

```json
{
  "setup": "s3-c",
  "ticket_id": "TKT-LOCAL-001",
  "customer_id": "CUS-1001",
  "customer_message": "Where is my order ORD-2026-00001?"
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

The model-facing tools are named by action, including `get_order_items`, `get_tracking_events`,
`create_return_request`, `request_refund`, and `cancel_order`. All writes use the same policy and review
gates. RAG and skills policy behavior stays in the established implementations; Langfuse and DeepEval
scoring are deferred.

## Rebuild the RAG index

After a policy change, refresh the skills files and the index from the PDFs in `data/policy/pdf/`:

```bash
.venv/bin/python scripts/sync_skills.py
.venv/bin/python scripts/build_rag_index.py
```

The retrieval evaluation is `notebooks/rag_evaluation.ipynb`, with the profiles in `configs/rag/`.
