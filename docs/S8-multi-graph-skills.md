# S8: supervisor + specialists · graph control · skills policy

Code: `setups/multi_graph.py` (shared with S7), part stages in `shared/flow.py`, supervisor and specialists in
`setups/multi_common.py`. Diagram: [S8-multi-graph-skills.drawio](S8-multi-graph-skills.drawio) (tabs: **Flow**,
**Data flow & access**). Owner: Lynn. Shared pipeline, gates and strategies: [README](README.md).

## At a glance

| | S8 |
|---|---|
| Agents | A supervisor and two specialists, `orders_shipping` and `returns_refunds` |
| Control | Graph: the supervisor only delegates. Each part runs S6's fixed stages, with the specialist in gather |
| Policy | Skills: each specialist's `policy` tool keeps by default to its own skill folder (`scope: own`) |
| Who writes | Code (each part's write stage), from the specialist's proposal |
| Who ends the ticket | Code: the strictest outcome of the parts, or a person if the supervisor gives up |
| Variants | `s8-a` … `s8-d` (supervisor × specialist gather strategy) |

## How S8 differs from its neighbours

| Compared with | What changes | What stays the same |
|---|---|---|
| [S7](S7-multi-graph-rag.md) | Policy: skills folders instead of RAG sections | Everything else: same code, same variants |
| [S6](S6-single-graph-skills.md) | Agents: one agent runs the stages once, with every read tool and every skill | Graph stages, skills, no agent writes |
| [S4](S4-multi-agentic-skills.md) | Control: specialists hold the write tools and `finish`; the supervisor holds `finish` | Supervisor + specialists, skills, same tool subsets and scoping |

## Variants

| Variant ID | Supervisor | Specialists (gather) |
|---|---|---|
| `s8-a` | react | react |
| `s8-b` | plan_execute | plan_execute |
| `s8-c` | react | plan_execute |
| `s8-d` | plan_execute | react |

## Architecture: who decides what

The same as S7:

- **The supervisor only delegates** (`ask_specialist`, no `finish`, no triage).
- **Each part runs the fixed stages** with the specialist in gather. Refusals go back to gather.
- **Code decides the outcome:** the strictest part, or a person if the supervisor gives up.

## Flow: a turn

See the **Flow** tab. The top row of the tab shows the stores behind the tools: Postgres for reads, and the policy path as two separate calls, `policy(open)` → the skill's SKILL.md, then `policy(read)` → a clause file it lists. `rules.json` feeds the write gate, and executed writes and the ticket land in Postgres. A specialist keeps by default to its own skill folder.

The same steps as S7:

1. The guard screens the new customer message.
2. The supervisor asks a specialist for a part, which runs gather → propose → check → review → write.
3. The part's report goes back. The supervisor asks for the next part or stops.
4. A handoff in any part ends the whole turn.
5. Otherwise the strictest outcome becomes the status. The shared reply writer, reply check and ticket save run.

## Data flow: what each LLM call sees

| LLM role | Sees | Does not see |
|---|---|---|
| Guard | The new customer message only | |
| Supervisor | The brief, and each part's report: outcome, note, writes | Any record, tool result or policy text |
| Specialist (gather) | The brief plus `your_task`, **its own** skills in the catalog up front, its own calls and results. After a refusal, also its own evidence and the `feedback` | The other folder's catalog (unless it lists with `scope: all`), the other specialist's evidence |
| Specialist (propose) | The brief plus `your_task`, and all evidence that specialist has gathered this turn, including the full text of files read | The other specialist's evidence |
| Approver (write) | The proposed write, its evidence, the rule check and `review_reason` | |
| Reply writer | The final status and note, all evidence from every part, problems code found | |

## Data access

| Component | Read tools | Policy | Writes |
|---|---|---|---|
| Supervisor | none | none | none |
| orders_shipping (gather) | `get_order`, `get_customer_orders`, `get_shipments`, `get_tracking_events`, `get_order_items` | skill folder `orders-and-shipping`; `scope: all` for the rest | none (proposes) |
| returns_refunds (gather) | `get_order`, `get_customer_orders`, `get_order_items`, `get_payments`, `get_returns_and_refunds`, `get_shipments` | skill folder `returns-and-refunds`; `scope: all` for the rest | none (proposes) |
| Part write stages (code) | `backend_write` (internal) | rules.json | returns + return_items, refunds (pending), orders/order_items/shipments (cancel), backend_writes |
| Ticket save (code) | none | | support_tickets |

**Skills scope.** With `scope: own`, `list`, `open` and `read` keep to the specialist's folder, and opening the
other folder is refused. This is stricter than S7, where the specialist sees every section name and the
search only filters hits.

## How the variants differ

The same mechanics as S7. See [S7: How the variants differ](S7-multi-graph-rag.md#how-the-variants-differ).

- **Supervisor react (`s8-a`, `s8-c`):** one part at a time; stops by not calling a tool. 3 identical errors mean it gives up.
- **Supervisor plan_execute (`s8-b`, `s8-d`):** every part planned up front. A completion check adds open parts.
  Giving up sends the ticket to a person.
- **Specialists react gather (`s8-a`, `s8-d`):** as `s6-a`. Each policy file is two steps.
- **Specialists plan_execute gather (`s8-b`, `s8-c`):** as `s6-b`. Their catalog lists only their own skills, so
  plans name files from their own folder.

## Worth knowing

- A proposal must cite a file or clause heading that was read this turn. Reading the other area's policy needs `scope: all`.
- Each `open` and `read` counts against the 40 tool calls, shared with every part's graph write attempts.
- A request neither specialist covers still goes to `orders_shipping`, which proposes awaiting_human.
