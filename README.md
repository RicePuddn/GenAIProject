# Policy-Bound Multi-Agent Customer Support Operations

IS469 Generative and Agentic AI group project comparing single-agent and multi-agent
customer support systems with RAG or skills-style policy retrieval.

The implemented application is a **single-agent LangGraph flow with RAG or skills-style policies**.
It supports ReAct and plan-and-execute using Groq's `openai/gpt-oss-120b`, with
`meta-llama/llama-prompt-guard-2-86m` screening incoming messages for prompt attacks.
Other system variants are still in progress.

## Install and configure

Run commands from the repository root. Requires Python 3.11–3.13, uv, and Docker
with Docker Compose. Start Docker Desktop before starting PostgreSQL.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,rag]"
cp .env.example .env
```

If `.env` already exists, update it directly. Set `GROQ_API_KEY` to your Groq key.
The default local database configuration is:

```dotenv
POSTGRES_PASSWORD=support_ops_local
POSTGRES_PORT=5432
SUPPORT_OPS_DATABASE_URL=postgresql://support_ops:support_ops_local@127.0.0.1:5432/support_ops
```

If you change the password or host port, update the database URL to match.
Changing the password in `.env` does not reset the password in an existing database volume.

## RAG

RAG uses the PDFs in `policy/raw_policy/pdf/`. Set `PINECONE_API_KEY` in `.env`.
Default models and parameters are in
`configs/base.yaml`; evaluation profiles are in `configs/rag/`.
The API preloads models at startup and verifies the uploaded corpus.

```python
from support_ops.policy.rag import retrieve

chunks = retrieve("Can I return a damaged final-sale item?")
```

After changing policies, regenerate embeddings and upload to the existing index:

```bash
.venv/bin/python -m support_ops.policy.rag.embeddings
.venv/bin/python -m support_ops.policy.rag.pinecone_store
```

For evaluation, open `src/support_ops/policy/rag/evaluation.ipynb`, select the `.venv`
kernel and run all cells. Detailed and summary JSON results go to `results/rag/`.

## Start the database and application

```bash
docker compose up -d --wait postgres
.venv/bin/python scripts/seed_backend.py
.venv/bin/uvicorn support_ops.api:app --reload --host 127.0.0.1 --port 8000
```

Seeding creates any missing tables and inserts fictional ecommerce records with
product names, customers, payments, tracking events, returns, refunds, and support tickets.
It preserves existing rows. See [database details](data/backend_seed/README.md) for
sample IDs and how to reset the local fixtures.

For subsequent runs, start PostgreSQL and Uvicorn again; reseeding is optional.
Use `Ctrl+C` to stop the API and `docker compose stop postgres` to stop the database.

## Try the chat endpoint

Open **[Swagger UI](http://127.0.0.1:8000/docs)** in your browser:

1. Expand **POST /chat** and click **Try it out**.
2. Paste a request body from below.
3. Click **Execute** and inspect the response body.

The application currently exposes an HTTP API; Swagger UI is its interactive interface.
[GET /health](http://127.0.0.1:8000/health) checks that the API is running. It does
not check database connectivity or Groq credentials.

For a shipment lookup:

```json
{
  "ticket_id": "TKT-LOCAL-001",
  "customer_message": "Where is my order ORD-2026-00001?",
  "strategy": "react"
}
```

For policy discovery and clause reading:

```json
{
  "ticket_id": "TKT-LOCAL-002",
  "customer_message": "Where is my order ORD-2026-00001? Can I return an unused item in its original packaging?",
  "strategy": "react"
}
```

The policy example should retrieve order/delivery facts and the returns policy.
Inspect `tool_calls` for `policy` inputs and outputs. RAG mode returns retrieved
section IDs and evidence; skills mode reads `SKILL.md` and supporting files.
Tool order can vary with the model. Return eligibility depends
on the workflow date: the API uses the current UTC date, while programmatic workflows
use `run.clock_start` unless a clock is injected. This order was delivered on
5 September 2026, so its eligibility changes with the date you run it.

You can also use curl or Postman:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"ticket_id":"TKT-LOCAL-001","customer_message":"Where is my order ORD-2026-00001?","strategy":"react"}'
```

### Multi-turn conversations

The API is stateless: the server keeps nothing between requests. To continue a ticket, copy
the `history` array from the previous response into the next request:

```json
{
  "ticket_id": "TKT-LOCAL-004",
  "customer_message": "It is ORD-2026-00001.",
  "history": [
    {"role": "customer", "content": "My order has not arrived."},
    {"role": "agent", "content": "Could you share your order number?"}
  ]
}
```

Earlier turns precede the new message in the agent's context; only the new message is
screened by Prompt Guard. History holds text turns only, never tool results, and the gateway
enforces its limits on every request, so edited history cannot authorise a write.

### Live steps: `/chat/stream`

`POST /chat/stream` takes the same body as `/chat` and answers with Server-Sent Events
(`text/event-stream`) while the turn runs: a `trace` event for each step as it is logged
(guard, thought, tool call, tool result, handoff), then one `result` event holding exactly
the `/chat` response, or an `error` event. It is a single stateless POST, the pattern of
MCP's Streamable HTTP transport. Steps stream whole, not token by token; `/chat` is unchanged
and remains the endpoint for scripts and evaluation.

### Demo page

A Streamlit page shows a chat window and, while the agent works, its steps as they stream in:
the guard result, each thought, each tool call with its result (retrieved policy clauses as a
table) and any handoff. Under each reply the full timeline stays available with cost and
timing. It keeps the conversation and sends it as `history`, and **New ticket** starts over.
With the API running:

```bash
uv pip install --python .venv/bin/python -e ".[demo]"
.venv/bin/streamlit run scripts/demo_app.py
```

## Compare the two strategies

| Setting | Behavior |
| --- | --- |
| `react` | The supervisor selects one tool, executes it, receives the result, then chooses the next step. |
| `plan_and_execute` | The supervisor produces an ordered plan using the shared tool schemas and available policy filenames. The workflow validates the entire plan and executes its calls sequentially. Invalid plans or recoverable tool errors trigger a bounded correction using completed results. |

Both strategies use the same model wrapper, tool contracts, policy files, mock approval
rule, tracing, and final response node. A successful plan executes without supervisor
calls between its tools. Plans use known literal arguments; they cannot substitute an
unknown result from a previous call into a later call.

Select the strategy in each request. For example, use the same message with:

```json
{
  "ticket_id": "TKT-LOCAL-003",
  "customer_message": "Where is my order ORD-2026-00001? Can I return an unused item in its original packaging?",
  "strategy": "plan_and_execute"
}
```

If you omit `strategy`, the application uses `agent.strategy` in
[configs/base.yaml](configs/base.yaml):

```yaml
agent:
  strategy: react  # react | plan_and_execute
```

Restart Uvicorn after editing configuration or `.env`; these are loaded once per
server process. Python source changes reload automatically with `--reload`.

## Read the response and trace

The `/chat` response includes:

- `reply`, `ticket_status`, and any `handoff_packet`.
- `evidence`: tool results, including errors retained for diagnosis.
- `tool_call_count`: tools that actually started execution, including failed attempts.
- `tool_calls`: requested tool names and inputs, linked to execution outputs and elapsed time.
  A plan stopped by an error or handoff can include later requested calls with `execution: null`.
- `trace`: LLM call count, token totals, configured cost estimate, elapsed seconds,
  tool count, and chronological node/LLM/tool/handoff events. `llm_response` events carry
  the model's `reasoning` when the provider returns it (gpt-oss on Groq does); it is logged
  for inspection and never sent back to the model.

Tool durations and total elapsed time use a monotonic timer. `simulated_at` records
the injected workflow clock, which stays fixed unless explicitly advanced.

LLM responses are cached in `results/cache` by default. Cached calls have
`cached: true` and add zero API cost, while token totals still report their recorded
usage. To compare live timings, set `llm.cache_dir: null` in `configs/base.yaml`
and restart the server. Prompt Guard caching is disabled independently.
Cost estimates depend on configured prices; unpriced models contribute zero.
The workflow bounds supervisor turns, tool executions, replans, and accumulated
LLM cost between calls. A single call can exceed the remaining cost budget.

## Policies and mock operations

The current tools are `read_record`, the order lookups `get_shipments_for_order`,
`get_shipment_events_for_order`, `get_items_for_order` (with product details),
`get_payments_for_order` and `get_returns_for_order` (returns and refunds), then `policy`
and `ticket_state`.
A dedicated final response node generates the customer reply through the shared LLM wrapper.

Policy retrieval follows `list` → `open` a skill's `SKILL.md` → `read` a relevant
clause. Available skills are `orders-and-shipping` and `returns-and-refunds`.
Policy checks and the ticket tool share [rules.json](src/support_ops/policy/rules/rules.json):
refund limits, return deadlines, cancellation conditions and human-review routes.
Checks require recorded facts; missing facts need human review. PDF fingerprints
flag policy changes that require a rules review. `policy.mode` in `configs/base.yaml` selects the
policy provider plugged into the `policy` tool.

Unsupported write operations and above-limit refunds require a human handoff.

PostgreSQL tools only read data. Approval tokens, mock writes, audits, and handoff
packets are held in memory for each request. A mock approval does not issue a
real refund or update PostgreSQL. Tokens are opaque simulation artifacts; signed
token validation and durable ticket state are future work. Conversation history
travels with each request (see above), but approvals from earlier requests are not restored.
Prompt Guard screens prompt attacks; it does not provide a complete abuse or
out-of-scope classification system. This local experiment API has no customer
authentication or order ownership checks.

## Automated checks

Install the development dependencies from the setup command, then run:

```bash
.venv/bin/python -m pytest -q
```

The regression suite uses the scripted LLM helper and isolated tool doubles, so it
needs neither a Groq key nor a running PostgreSQL instance. Live checks use Swagger
UI or curl as described above.

## Layout

```text
configs/                      models, strategy, simulation clock and limits
policy/skills/                policy entrypoints and clause files
policy/raw_policy/            source policy PDFs
policy/chunk_policy/          section chunks with reference IDs
policy/embed_chunk/           saved vectors and upload manifest
data/backend_seed/schema.sql  PostgreSQL schema
scripts/seed_backend.py       deterministic ecommerce fixtures
src/support_ops/
  api.py                      FastAPI request/response boundary
  helpers/                    shared LLM wrapper, clock, config and tracing
  backend/postgres.py         allowlisted database reads
  policy/provider.py          contract each policy provider implements (RAG/skills)
  policy/rules/               grouped JSON rules and deterministic checks
  policy/skills.py            skill navigation (the skills provider)
  policy/rag/                 policy chunking, embeddings, retrieval and evaluation
  gateway/ticket_state.py     request-local approvals, audit and handoff
  systems/tools.py            shared tool schemas, validation and dispatcher
  systems/single_agent.py     LangGraph nodes and strategy routing
  agents/, supervisor/, eval/ reserved for other variants and evaluation
results/                      local traces and response cache
tests/                       helper and workflow regression tests
```
