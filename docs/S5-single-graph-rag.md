# S5: single agent · graph control · RAG policy

Code: `setups/single_graph.py` (shared with S6), stages in `shared/flow.py`. Diagram:
[S5-single-graph-rag.drawio](S5-single-graph-rag.drawio) (tabs: **Flow**, **Data flow & access**). Owner: FP.
Shared pipeline, gates and strategies: [README](README.md).

## At a glance

| | S5 |
|---|---|
| Agents | One agent, used only in the gather stage and for the proposal |
| Control | Graph: a fixed LangGraph sequence decided in code, gather → propose → rule check → review → write |
| Policy | RAG: `policy(action="search")` over the Pinecone index |
| Who writes | Code (the graph's write stage), from the agent's proposal |
| Who ends the ticket | Code: the proposal's outcome, unless something hands it off |
| Variants | `s5-a` (react gather), `s5-b` (plan_execute gather) |

## How S5 differs from its neighbours

| Compared with | What changes | What stays the same |
|---|---|---|
| [S6](S6-single-graph-skills.md) | Policy: skills files instead of RAG search | Everything else: same code, same variants |
| [S7](S7-multi-graph-rag.md) | Agents: a supervisor asks specialists, and each part runs these stages separately. The ticket takes the strictest part | Graph stages, RAG, no agent writes |
| [S1](S1-single-agentic-rag.md) | Control: the agent holds the write tools, asks for review and ends the ticket itself | One agent, RAG, the same read tools |

## Variants

| Variant ID | Gather-stage strategy |
|---|---|
| `s5-a` | react |
| `s5-b` | plan_execute |

The strategy applies only to gather. Propose, check, review and write are the same code in both.

## Architecture: who decides what

| The agent decides | Code decides |
|---|---|
| Which records and policy to read (gather) | When gathering ends and proposing starts |
| The proposal: outcome, note, writes, `route`, `policy_exception`, `review_reason` | Whether each write is allowed, reviewed and executed, in order |
| | Sending a refusal back to gather, with the reason |
| | Ending the turn on a handoff |
| | The reply, its checks and the ticket save |

The agent has no write tools, no `request_review` and no `finish`.

## Flow: a turn

See the **Flow** tab. The top row of the tab shows the stores behind the tools: Postgres for reads, and the policy path `policy(search)` → hybrid retriever (Nomic dense + BM25, fused by RRF, bge rerank, top 4 + referenced clauses) → Pinecone index. `rules.json` feeds the write gate, and executed writes and the ticket land in Postgres.

1. The guard screens the new customer message.
2. **Gather.** The agent reads by its strategy with the seven read tools and `policy`. Gathering ends when it
   stops asking for tools.
3. **Propose.** One structured output, from the same agent prompt plus the propose instructions: `outcome`
   (resolved, awaiting_customer or awaiting_human), `note`, `writes` and `review_reason`.
4. For each proposed write, in order:
   - **Rule check** (the write gate). A write this ticket already made is replayed, not repeated.
   - **Review** (approver LLM), on the approver route only.
   - **Write** (executor), recorded as `backend_write` in the trace.
5. Anything refused goes back to gather with the reason (`feedback`): a block, a citation that wasn't
   retrieved, unseen returns or refunds, an invalid proposed write, or a fixable review rejection.
6. A handoff (the write gate, a second rejection, a proposal of awaiting_human, or a failed gather) ends the turn.
7. Otherwise the proposal's outcome is the ticket's status. The shared reply writer, reply check and ticket save run.

## Data flow: what each LLM call sees

| LLM role | Sees | Does not see |
|---|---|---|
| Guard | The new customer message only | |
| Agent (gather) | The brief, the policy section names up front, its own calls and results. After a refusal, also its earlier evidence and the `feedback` | Write tools: it cannot see a write's result until the graph refuses one |
| Agent (propose) | The brief, all its own evidence, any `feedback` | |
| Approver (write) | The proposed write, its evidence, the rule check and `review_reason` | |
| Reply writer | The status, the proposal's note, all evidence, problems code found | |

## Data access

| Component | Tools | Reads | Writes |
|---|---|---|---|
| Agent (gather) | the seven read tools | orders, order_items, products, shipments, shipment_items, shipment_events, payments, returns, return_items, refunds | none |
| Agent (gather) | `policy` (`search`, `check`) | Pinecone index; rules.json for `check` | none |
| Graph write stages (code) | `backend_write` (internal, not a model tool) | orders, order_items, shipments, payments, returns/refunds, rules.json | returns + return_items, refunds (pending), orders/order_items/shipments (cancel), backend_writes |
| Ticket save (code) | none | | support_tickets |

All reads and writes are scoped to the signed-in account. Each graph write attempt counts against the 40 tool calls.

## How the variants differ

**`s5-a`: react gather.** One lookup per step. Gathering ends when the agent replies with a one-line summary
and no tool call. There is no nudge, since there is no `finish`. The same tool failing with the same error 3
times in a row ends gathering, and the turn goes to a person with no proposal.

**`s5-b`: plan_execute gather.** It plans its lookups as JSON. Gathering ends on an empty plan (`{"calls":
[]}`), usually at the first completion check. With no writes in its toolset, a plan never meets `blocked` or
`review_required`, so replans come only from failed lookups or invalid plans. No working plan after 2 replans,
or still planning after 3 checks, ends the turn with a person.

**Both:** after a refusal, gather starts its loop again with its earlier evidence and the `feedback`. A
plan_execute agent is told not to repeat calls already in the evidence. Each refusal costs at least one more
gather step and one propose call. Refusals are limited by the budget and the review limit, not by `max_replans`.

## Worth knowing

- A proposal of `awaiting_human` ends the turn with nothing written.
- A message unrelated to store support: nothing is gathered, the proposal is `resolved` with no writes, and the
  reply invites a support question.
- A citation that gather never retrieved costs a full gather → propose loop, not one extra call.
