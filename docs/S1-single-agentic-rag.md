# S1: single agent · agentic control · RAG policy

Code: `setups/single_agentic.py` (shared with S2). Diagram: [S1-single-agentic-rag.drawio](S1-single-agentic-rag.drawio)
(tabs: **Flow**, **Data flow & access**). Owner: FP. Shared pipeline, gates and strategies: [README](README.md).

## At a glance

| | S1 |
|---|---|
| Agents | One agent holds every tool |
| Control | Agentic: the agent decides what to read, when to write, when to ask for review, and when it is finished |
| Policy | RAG: `policy(action="search")` over the Pinecone index of the two policy PDFs |
| Who writes | The agent, through the named write tools |
| Who ends the ticket | The agent, with `finish` |
| Variants | `s1-a` (react), `s1-b` (plan_execute) |

## How S1 differs from its neighbours

Each neighbour changes exactly one thing.

| Compared with | What changes | What stays the same |
|---|---|---|
| [S2](S2-single-agentic-skills.md) | Policy: skills files (`open`, then `read`) instead of RAG search | Everything else: same code, same variants |
| [S3](S3-multi-agentic-rag.md) | Agents: a supervisor delegates to two specialists, each with a subset of the tools and a scoped policy | Agentic control, RAG, named writes, `finish` |
| [S5](S5-single-graph-rag.md) | Control: the agent only gathers. A fixed graph proposes, checks, reviews and writes | One agent, RAG, the same read tools |

## Variants

| Variant ID | Agent strategy |
|---|---|
| `s1-a` | react |
| `s1-b` | plan_execute |

## Architecture: who decides what

| The agent decides | Code decides |
|---|---|
| Which records and policy to read | Whether a write is allowed (rules on the recorded facts) |
| Whether and what to write (`create_return_request`, `request_refund`, `cancel_order`) | Whether the citations were retrieved this turn |
| When to ask for review (`request_review`) | Whether the write needs a review (rules' route, made stricter by the agent's `route` or `policy_exception`) |
| How the turn ends (`finish`: resolved, awaiting_customer, awaiting_human, with a note) | The review verdict (a separate approver call) |
| | The reply, its checks and the ticket save |

## Flow: a turn

See the **Flow** tab. The top row of the tab shows the stores behind the tools: Postgres for reads, and the policy path `policy(search)` → hybrid retriever (Nomic dense + BM25, fused by RRF, bge rerank, top 4 + referenced clauses) → Pinecone index. `rules.json` feeds the write gate, and executed writes and the ticket land in Postgres.

1. The guard screens the new customer message. A flagged message gets the hold reply, and no agent runs.
2. The agent works by its strategy with the seven read tools, `policy`, the three named write tools,
   `request_review` and `finish`.
3. A named write tool goes through the write gate and returns one of:
   - `executed`;
   - `blocked` (the rules refused it, or the order's returns and refunds were not looked up yet, which are listed once);
   - `review_required` (approver route: call `request_review` with the same write, then write again);
   - `awaiting_human` (the turn ends).
4. The agent ends with `finish`. If it stops without calling it, or its strategy gives up, the turn goes to a person.
5. The shared reply writer, reply check and ticket save run.

## Data flow: what each LLM call sees

See the **Data flow & access** tab.

| LLM role | Sees | Does not see |
|---|---|---|
| Guard | The new customer message only | History, records, policy |
| Agent | The brief (message, history, signed-in account), the policy section names up front, and its own tool calls and results | Nothing is hidden from it: it holds every tool |
| Approver (write) | The proposed write, its evidence, the rule check and `review_reason` (treated as an unverified claim) | The agent's reasoning |
| Reply writer | The status, the agent's `finish` note, all tool evidence, problems code found | Internal IDs it must not repeat (checked in code) |
| Approver (reply) | The draft reply, the ticket and the evidence | |

## Data access

Every read and write goes through the account-scoped reader and writer. Another customer's records read as
missing.

| Component | Tools | Reads | Writes |
|---|---|---|---|
| Agent | `get_customer_orders`, `get_order`, `get_order_items`, `get_shipments`, `get_tracking_events`, `get_payments`, `get_returns_and_refunds` | orders, order_items, products, shipments, shipment_items, shipment_events, payments, returns, return_items, refunds | none directly |
| Agent | `policy` (`search`, `check`) | Pinecone index (hybrid dense + BM25, reranked, top 4 + referenced clauses); rules.json for `check` | none |
| Agent | `create_return_request`, `request_refund`, `cancel_order`, `request_review` | through the write gate: orders, order_items, shipments, payments, returns/refunds, rules.json | returns + return_items, refunds (pending), orders/order_items/shipments (cancel), backend_writes |
| Ticket save (code) | none | | support_tickets |

## How the variants differ

The tools, gates and prompt are identical. Only the loop around the agent changes.

**`s1-a`: react.** One call, see the result, pick the next.

- A typical refund: `get_order` → `get_payments` → `get_returns_and_refunds` → `policy` → `request_refund` →
  `finish`. Each one is a separate LLM step.
- An approver-route write takes three steps: the write (`review_required`), `request_review`, then the same write again.
- A `blocked` result is just the next thing it sees, so it can correct the write or explain the policy. Only
  the budget limits how often.
- If it replies without a tool call, it is nudged once to call `finish`. A second time hands the turn off.

**`s1-b`: plan_execute.** Plan a list of calls as JSON, run it, then get asked again.

- Usually two rounds. The first plan fetches facts and policy: it cannot cite clause IDs it hasn't seen. The
  completion check then plans the write and `finish`.
- An approver-route write stops the plan at `review_required`. The next plan is `request_review` → the write
  → `finish`, and that uses one of the 2 replans.
- `blocked` also stops the plan and uses a replan, so this variant has less room to correct a write than `s1-a`.
- The completion check prompt itself asks for anything still owed, "such as a refund owed after a
  cancellation". `s1-a` gets no such prompt.
- There is no nudge. A plan that never reaches `finish` within 3 checks ends the run, and the turn goes to a person.

## Worth knowing

- A write requiring review never runs unless `request_review` approved that exact write (the same operation,
  order, amount and items).
- A repeated write returns its first result; nothing is written twice.
- The single agent's prompt does not tell it to refund a cancelled order's captured balance. The rules allow
  that refund (ODR-OrderChangeCancel-4.2), but only `s1-b`'s completion check prompts for it.
- RAG search returns clauses with IDs, and a write cites those IDs. One search usually answers one policy
  question.
