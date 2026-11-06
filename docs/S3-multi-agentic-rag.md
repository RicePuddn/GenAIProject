# S3: supervisor + specialists · agentic control · RAG policy

Code: `setups/multi_agentic.py` (shared with S4), supervisor and specialists in `setups/multi_common.py`.
Diagram: [S3-multi-agentic-rag.drawio](S3-multi-agentic-rag.drawio) (tabs: **Flow**, **Data flow & access**).
Owner: FT. Shared pipeline, gates and strategies: [README](README.md).

## At a glance

| | S3 |
|---|---|
| Agents | A supervisor and two specialists, `orders_shipping` and `returns_refunds` |
| Control | Agentic: the supervisor decides who handles each part. Each specialist decides its own reads, writes and reviews |
| Policy | RAG search, kept by default to the specialist's own section (`scope: own`) |
| Who writes | The specialists, through the named write tools |
| Who ends the ticket | The supervisor, with `finish`, but never milder than the strictest specialist report |
| Variants | `s3-a` … `s3-d` (supervisor × specialist strategy) |

## How S3 differs from its neighbours

| Compared with | What changes | What stays the same |
|---|---|---|
| [S4](S4-multi-agentic-skills.md) | Policy: skills folders instead of RAG sections | Everything else: same code, same variants |
| [S1](S1-single-agentic-rag.md) | Agents: one agent with every tool instead of a supervisor and two scoped specialists | Agentic control, RAG, named writes, `finish` |
| [S7](S7-multi-graph-rag.md) | Control: specialists only gather inside fixed graph stages, and the supervisor has no `finish` (code takes the strictest part) | Supervisor + specialists, RAG, same tool subsets and scoping |

## Variants

| Variant ID | Supervisor | Specialists |
|---|---|---|
| `s3-a` | react | react |
| `s3-b` | plan_execute | plan_execute |
| `s3-c` | react | plan_execute |
| `s3-d` | plan_execute | react |

## Architecture: who decides what

- **The supervisor** decides which specialist handles each part, and how the ticket ends. It has
  `ask_specialist(specialist, request)` and `finish`. It never sees records or policy, only each specialist's report.
- **Each specialist** acts on its part as S1's agent does, then finishes with `finish`. Its report (outcome,
  note, writes made) goes back to the supervisor.
- **Code sets a floor on the outcome.** The supervisor cannot end the ticket milder than the strictest
  report (awaiting_human, then awaiting_customer, then resolved). The trace shows `outcome_raised` when it does.
- **Code decides every write**, through the same write gate as every system.

## Flow: a turn

See the **Flow** tab. The top row of the tab shows the stores behind the tools: Postgres for reads, and the policy path `policy(search)` → hybrid retriever (Nomic dense + BM25, fused by RRF, bge rerank, top 4 + referenced clauses) → Pinecone index. `rules.json` feeds the write gate, and executed writes and the ticket land in Postgres. A specialist's search keeps by default to its own section's hits.

1. The guard screens the new customer message.
2. The supervisor asks a specialist for each part. Each ask runs the specialist fresh, with the brief plus `your_task`.
3. A specialist's writes go through the write gate, with `request_review` for the approver route. A write
   handed to a person ends the whole turn.
4. The specialist ends with `finish`, and its report goes back to the supervisor.
5. The supervisor's `finish` ends the turn, raised to the strictest part's outcome if needed.
6. The shared reply writer, reply check and ticket save run.

## Data flow: what each LLM call sees

See the **Data flow & access** tab.

| LLM role | Sees | Does not see |
|---|---|---|
| Guard | The new customer message only | |
| Supervisor | The brief, and each report: specialist, outcome, note, writes (operation, order, amount, items), plus an `error` if the specialist failed | Any record, tool result or policy text |
| Specialist | The brief plus `your_task`, every policy section name up front, its own tool calls and results in this ask | The other specialist's work, and its own earlier asks (each ask starts fresh) |
| Approver (write) | The proposed write, its evidence, the rule check and `review_reason` | |
| Reply writer | The final status and note, **all** tool evidence from every specialist, problems code found | |

## Data access

| Component | Read tools | Policy | Writes |
|---|---|---|---|
| Supervisor | none | none | none |
| orders_shipping | `get_order`, `get_customer_orders`, `get_shipments`, `get_tracking_events`, `get_order_items` | RAG search, own section "Orders and shipping"; `scope: all` for the rest | named writes + `request_review` (by prompt: `cancel_order`) |
| returns_refunds | `get_order`, `get_customer_orders`, `get_order_items`, `get_payments`, `get_returns_and_refunds`, `get_shipments` | RAG search, own section "Returns and refunds"; `scope: all` for the rest | named writes + `request_review` (by prompt: `create_return_request`, `request_refund`) |

Tables behind them: the same as S1. Only `orders_shipping` can see tracking events, and only `returns_refunds`
can see payments and returns. Write operations are not limited per specialist; only the prompts say whose
they are. All access is scoped to the signed-in account.

**RAG scope.** A `scope: own` search keeps only hits from the specialist's section and the clauses they
reference. If none match, the result says to search again with `scope: all`. The specialist still sees every
section name up front.

## How the variants differ

**Supervisor react (`s3-a`, `s3-c`).** It asks one part, reads the report, then asks the other specialist, asks
the same one again with what was missing, or calls `finish`. A cancellation becomes orders_shipping
(`cancel_order`), then, with that report in view, returns_refunds refunds the captured balance. If it stops
without `finish`, it is nudged once, then handed off.

**Supervisor plan_execute (`s3-b`, `s3-d`).** It plans every `ask_specialist` from the message alone. The
cancellation case is usually planned as two asks, because the supervisor prompt says a cancellation needs
both. A missing refund ask is added by the completion check ("the refund a cancelled order is owed"). A
report carrying an `error` stops the plan and uses one of the 2 replans. A report that ends `awaiting_human`
does not stop it.

**Specialists react (`s3-a`, `s3-d`).** They behave as `s1-a`: one call per step, write →
`request_review` → write, and a nudge if they stop without `finish`.

**Specialists plan_execute (`s3-b`, `s3-c`).** They behave as `s1-b`: a facts-and-policy plan, then a
completion check that plans the write and `finish`. A specialist with no working plan, or still unfinished
after 3 checks, reports an `error` to the supervisor.

## Worth knowing

- A specialist that stops without `finish`, or gives up, reports outcome `awaiting_human` with an error. The
  supervisor can ask again, but the failed part stays in the parts, so the outcome floor ends the ticket
  `awaiting_human` even if the retry succeeds.
- Re-asking the same specialist repeats its lookups unless the supervisor's `request` says what is already known.
- The ticket's 30 agent steps count every supervisor decision and every specialist step.
- A specialist's invalid-argument count lasts the whole turn: a second invalid call stops the turn, even
  across two asks.
