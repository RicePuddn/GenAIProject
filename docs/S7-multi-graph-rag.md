# S7: supervisor + specialists · graph control · RAG policy

Code: `setups/multi_graph.py` (shared with S8), part stages in `shared/flow.py`, supervisor and specialists in
`setups/multi_common.py`. Diagram: [S7-multi-graph-rag.drawio](S7-multi-graph-rag.drawio) (tabs: **Flow**,
**Data flow & access**). Owner: FT. Shared pipeline, gates and strategies: [README](README.md).

## At a glance

| | S7 |
|---|---|
| Agents | A supervisor and two specialists, `orders_shipping` and `returns_refunds` |
| Control | Graph: the supervisor only delegates. Each part runs S5's fixed stages, with the specialist in gather |
| Policy | RAG search, kept by default to the specialist's own section (`scope: own`) |
| Who writes | Code (each part's write stage), from the specialist's proposal |
| Who ends the ticket | Code: the strictest outcome of the parts, or a person if the supervisor gives up |
| Variants | `s7-a` … `s7-d` (supervisor × specialist gather strategy) |

## How S7 differs from its neighbours

| Compared with | What changes | What stays the same |
|---|---|---|
| [S8](S8-multi-graph-skills.md) | Policy: skills folders instead of RAG sections | Everything else: same code, same variants |
| [S5](S5-single-graph-rag.md) | Agents: one agent runs the stages once, with every read tool | Graph stages, RAG, no agent writes |
| [S3](S3-multi-agentic-rag.md) | Control: specialists hold the write tools and `finish`; the supervisor holds `finish` | Supervisor + specialists, RAG, same tool subsets and scoping |

## Variants

| Variant ID | Supervisor | Specialists (gather) |
|---|---|---|
| `s7-a` | react | react |
| `s7-b` | plan_execute | plan_execute |
| `s7-c` | react | plan_execute |
| `s7-d` | plan_execute | react |

## Architecture: who decides what

- **The supervisor only delegates.** Its one tool is `ask_specialist(specialist, request)`. It does not
  classify or redirect the message (no triage), write, or end the ticket, and it has no `finish`.
- **Each part runs the fixed stages**: gather (the specialist's reads and scoped policy) → propose → rule check
  → review (approver route) → write. Refusals go back to gather with the reason.
- **Code decides the outcome:** the strictest of the parts (awaiting_human, then awaiting_customer, then
  resolved). If the supervisor gives up, the ticket goes to a person, since a part may never have been asked.

## Flow: a turn

See the **Flow** tab. The top row of the tab shows the stores behind the tools: Postgres for reads, and the policy path `policy(search)` → hybrid retriever (Nomic dense + BM25, fused by RRF, bge rerank, top 4 + referenced clauses) → Pinecone index. `rules.json` feeds the write gate, and executed writes and the ticket land in Postgres. A specialist's search keeps by default to its own section's hits.

1. The guard screens the new customer message.
2. The supervisor asks a specialist for a part. That part runs gather → propose → check → review → write.
3. The part's report (outcome, note, writes) goes back to the supervisor, which asks for the next part or stops.
4. A handoff in any part (a failed gather, a proposal of awaiting_human, or the write gate) ends the whole turn.
5. When the supervisor stops asking, the strictest outcome becomes the status.
6. The shared reply writer, reply check and ticket save run.

## Data flow: what each LLM call sees

| LLM role | Sees | Does not see |
|---|---|---|
| Guard | The new customer message only | |
| Supervisor | The brief, and each part's report: outcome, note, writes (operation, order, amount, items) | Any record, tool result or policy text |
| Specialist (gather) | The brief plus `your_task`, every policy section name up front, its own calls and results. After a refusal, also its own evidence and the `feedback` | The other specialist's evidence |
| Specialist (propose) | The brief plus `your_task`, and **all evidence that specialist has gathered this turn**, earlier parts included | The other specialist's evidence |
| Approver (write) | The proposed write, its evidence, the rule check and `review_reason` | |
| Reply writer | The final status and note, all evidence from every part, problems code found | |

## Data access

| Component | Read tools | Policy | Writes |
|---|---|---|---|
| Supervisor | none | none | none |
| orders_shipping (gather) | `get_order`, `get_customer_orders`, `get_shipments`, `get_tracking_events`, `get_order_items` | RAG search, own section "Orders and shipping"; `scope: all` for the rest | none (proposes) |
| returns_refunds (gather) | `get_order`, `get_customer_orders`, `get_order_items`, `get_payments`, `get_returns_and_refunds`, `get_shipments` | RAG search, own section "Returns and refunds"; `scope: all` for the rest | none (proposes) |
| Part write stages (code) | `backend_write` (internal) | rules.json | returns + return_items, refunds (pending), orders/order_items/shipments (cancel), backend_writes |
| Ticket save (code) | none | | support_tickets |

The tables behind the read tools and the write gate's fact lookups are the same as S5's. All access is scoped
to the signed-in account.

## How the variants differ

**Supervisor react (`s7-a`, `s7-c`).** It asks one part, sees its report, and decides on the next. It stops by
replying without a tool call; there is no `finish` and no nudge. The same `ask_specialist` error 3 times in a
row means it gives up, and the ticket goes to a person.

**Supervisor plan_execute (`s7-b`, `s7-d`).** It plans every part from the message alone. A part's report
never stops the plan, because graph parts report an outcome, not an error. Only a handoff, which ends the
turn, cuts it short. The completion check plans the parts still open, such as the refund a cancelled order is
owed. An empty plan ends delegation. No readable plan, more than 2 replans, or still planning after 3 checks
sends the ticket to a person, even if every part that ran was resolved.

**Specialists react gather (`s7-a`, `s7-d`).** As `s5-a`: one lookup per step, ending when the specialist stops
calling tools. 3 identical errors end the turn with a person.

**Specialists plan_execute gather (`s7-b`, `s7-c`).** As `s5-b`: planned lookups, ending on an empty plan. No
working plan after 2 replans, or 3 checks, ends the turn with a person.

## Worth knowing

- If the supervisor asks no specialist (an off-topic message), the ticket is resolved and the reply says what
  support covers.
- A support request neither specialist covers (an account change, or asking for a person) still goes to
  `orders_shipping`, which proposes awaiting_human. The supervisor has no way to hand off itself.
- Unlike S3, a second ask to the same specialist can build on its first part: propose sees all that
  specialist's evidence from the turn.
- Every supervisor decision, gather step and propose call counts against the one 30-step budget.
