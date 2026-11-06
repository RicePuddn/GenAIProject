# Architecture

The application entrypoints and model-facing contracts live in the root-level `SupportOpsBot/` package.
The `SupportOpsBot/runtime/` package implements the shared eight-setup workflow, rules, database access,
and write/review safeguards. Setup IDs and policy assets remain under `/configs` and `/data`.

## Application package

| Path | Responsibility |
|---|---|
| `SupportOpsBot/chatRoute.py` | FastAPI app. Preserves `POST /chat`, `POST /chat/stream`, `GET /info`, and `GET /customers`. |
| `SupportOpsBot/demo.py` | Streamlit chat and live trace timeline. |
| `SupportOpsBot/services/structure/structure.py` | Strict tool input/output contracts, typed errors, and graph workflow models. |
| `SupportOpsBot/services/structure/prompts.py` | All role and workflow prompts, including guard and final reply. |
| `SupportOpsBot/services/tools/read/` | Account-scoped record reads and policy search/check tools. |
| `SupportOpsBot/services/tools/write/` | Return, refund, cancellation, and independent review tools. |
| `SupportOpsBot/services/tools/` | Tool contract and agent-control tools such as `finish` and `ask_specialist`. |
| `SupportOpsBot/rag/`, `SupportOpsBot/skills/` | Thin adapters to the existing policy providers and assets. |
| `SupportOpsBot/services/helpers/` | Trajectory persistence and operational summaries. |

The API is stateless. Clients pass the prior `history` with each turn. `/chat/stream` emits a `trace` event
for each live event, then a `result` or `error`. Every completed API turn also writes its request, response,
trace, and operational summary under `results/runs/` for future offline analysis. Langfuse and DeepEval
scoring are outside this rebuild.

## One turn

All setup variants use the same outer workflow:

```text
guard ─► setup body ─► final reply ─► wording checks ─► reply review ─► ticket save
  │          │                                                │
  └─ flagged ┴─ budget or provider failure ─► hold reply + handoff
```

The four retained setup implementations cover S1–S8. `/configs/setups.yaml` defines which agent count,
control style, policy mode, and strategy each setup ID selects.

| Setups | Runtime | Control |
|---|---|---|
| S1–S2 | `SupportOpsBot/runtime/setups/single_agentic.py` | One agent chooses among read, policy, write, review, and finish tools. |
| S3–S4 | `SupportOpsBot/runtime/setups/multi_agentic.py` | A supervisor delegates parts; each specialist chooses its tools and actions. |
| S5–S6 | `SupportOpsBot/runtime/setups/single_graph.py` | An agent gathers; code proposes, checks, reviews, and executes writes. |
| S7–S8 | `SupportOpsBot/runtime/setups/multi_graph.py` | Specialists gather within the same graph stages; the supervisor delegates. |

## Tool and write behavior

Each named tool is declared in its own module. The registry validates arguments before calling a handler and
validates each result before returning it to the model. A first invalid argument response lists the fields to
correct; a second invalid call stops that agent run. Invalid handler output is returned as a typed,
non-retryable error, and the same failing call cannot be repeated.

The model-facing tools are `get_customer_orders`, `get_order`, `get_order_items`, `get_shipments`,
`get_tracking_events`, `get_payments`, `get_returns_and_refunds`, `policy`, `create_return_request`,
`request_refund`, `cancel_order`, `request_review`, `finish`, and `ask_specialist`. There is no generic
`read_record` tool. Graph stages use the same internal write gate and executor as the named agentic write
tools.

The shared runtime retains the existing account scoping, policy rules, citation checks, review gates,
idempotency keys, retry policy, and per-ticket budgets. A write requiring approval remains blocked until
the independent reviewer approves the same operation and values. Every read and write uses the request's
signed-in customer scope when one is supplied.

## Policy, traces, and configuration

`SupportOpsBot/rag/` and `SupportOpsBot/skills/` expose the existing `RagPolicy` and `SkillsPolicy`; the
retrieval and skill behavior and `/data/policy/` assets are unchanged. `scripts/build_rag_index.py` and
`scripts/sync_skills.py` still use the established implementations.

The shared runtime records every tool, LLM, gate, review, and handoff event in a per-ticket `Tracer`. The
same event stream feeds `/chat/stream` and the Streamlit timeline. Run records are saved at
`results/runs/<UTC timestamp>-<run id>.json`.

`configs/base.yaml` holds shared values. `configs/setups.yaml` resolves an ID such as `s3-c` to its agent
count, control style, policy mode, and strategy. `configs/rag/` retains the retrieval profiles.

## Run entrypoints

```bash
.venv/bin/uvicorn SupportOpsBot.chatRoute:app --reload --host 127.0.0.1 --port 8000
.venv/bin/streamlit run SupportOpsBot/demo.py
```

The project package includes `SupportOpsBot`, including its runtime package so installed and
repository-root entrypoints can import the same implementation.
