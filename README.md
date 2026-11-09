# Policy-Bound Customer Support Operations

IS469 Generative and Agentic AI group project: eight support setups (single or multi-agent, agentic or
graph control, RAG or skills policy), 24 variants in all, compared on the same tickets. How the code
fits together is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Install

Requires Python 3.11–3.13, uv, and Docker with Docker Compose. Run commands from the repository root.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[rag,demo,evaluation]"
cp .env.example .env
```

The `evaluation` extra is needed only to run and analyse the evaluation (see below).

Set `OPENROUTER_API_KEY` for the default LLM provider (see the next section) and `PINECONE_API_KEY`
for the RAG setups (S1, S3, S5, S7). The default local database settings are:

```dotenv
POSTGRES_PASSWORD=support_ops_local
POSTGRES_PORT=5432
SUPPORT_OPS_DATABASE_URL=postgresql://support_ops:support_ops_local@127.0.0.1:5432/support_ops
```

If you change the password or port, update the URL to match. Changing the password in `.env` does not
reset the password of an existing database volume.

## Configure the LLM provider

The default profile uses OpenRouter's OpenAI-compatible API with `openai/gpt-oss-120b` for every role.
Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` to a key from your OpenRouter account.
The provider endpoint is `https://openrouter.ai/api/v1`.

To use local Ollama instead, set `SUPPORT_OPS_CONFIG=configs/ollama.yaml` in `.env` and start Ollama
with the model named in that profile. Restart the API after changing provider or model settings.

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
- `clock_start`: the workflow date defaults to `run.clock_start` (1 October 2026). Send a
  timezone-aware date to override it, e.g. to check return windows against today. The evaluation
  cases set their own date in `deepeval/cases/`.

`POST /chat/stream` takes the same body and answers with Server-Sent Events: a `trace` event per step,
then one `result` event holding the `/chat` response, or an `error` event.

The model-facing tools are named by action, including `get_order_items`, `get_tracking_events`,
`create_return_request`, `request_refund`, and `cancel_order`. All writes use the same policy and review
gates.

## Rebuild the RAG index

After a policy change, refresh the skills files and the index from the PDFs in `data/policy/pdf/`:

```bash
.venv/bin/python scripts/sync_skills.py
.venv/bin/python scripts/build_rag_index.py
```

The retrieval evaluation, which compares vector, hybrid and hybrid + rerank search, is
[results/rag/rag_evaluation.ipynb](results/rag/rag_evaluation.ipynb), with the profiles in `configs/rag/`.

## Run the evaluation

The evaluation sends 13 test cases to all 24 variants (312 runs), then scores the saved runs with
DeepEval, using `openai/gpt-oss-120b` as the judge. Running and scoring are separate steps. Start the
database and the API first, and use a dedicated evaluation database: collection resets the test tickets'
records before each run.

```bash
bash deepeval/collect_full.sh --dry-run   # preview the selection
bash deepeval/collect_full.sh             # collect; prints the output folder and the scoring command
```

```bash
run_dir="results/deepeval/data/raw/baseline/run1"   # the folder collection printed
.venv/bin/python deepeval/evaluate.py \
  --input "$run_dir/results.jsonl" \
  --output "$run_dir/scored.csv" \
  --judge-model openai/gpt-oss-120b
```

Add `--variant paraphrased` to collect the reworded questions for the robustness check. The test cases
and their expected answers, ticket statuses and tool calls are in `deepeval/cases/`. See
[deepeval/README.md](deepeval/README.md) for single-case runs, re-scoring and retries.

The analysis notebooks in `results/deepeval/analysis/` (consistency, robustness, scorecard, hypothesis
tests and findings) read the scored runs and write the report's tables and figures. Run them in order;
see [results/deepeval/README.md](results/deepeval/README.md).
