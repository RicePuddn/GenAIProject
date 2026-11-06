# S2: single agent · agentic control · skills policy

Code: `setups/single_agentic.py` (shared with S1). Diagram: [S2-single-agentic-skills.drawio](S2-single-agentic-skills.drawio)
(tabs: **Flow**, **Data flow & access**). Owner: Ernest. Shared pipeline, gates and strategies: [README](README.md).

## At a glance

| | S2 |
|---|---|
| Agents | One agent holds every tool |
| Control | Agentic: the agent decides what to read, when to write, when to ask for review, and when it is finished |
| Policy | Skills: `policy(action="open")` reads a skill's SKILL.md, `policy(action="read")` reads a clause file under `data/policy/skills/` |
| Who writes | The agent, through the named write tools |
| Who ends the ticket | The agent, with `finish` |
| Variants | `s2-a` (react), `s2-b` (plan_execute) |

## How S2 differs from its neighbours

| Compared with | What changes | What stays the same |
|---|---|---|
| [S1](S1-single-agentic-rag.md) | Policy: skills files instead of RAG search | Everything else: same code, same variants |
| [S4](S4-multi-agentic-skills.md) | Agents: a supervisor delegates to two specialists, each kept to its own skill folder | Agentic control, skills, named writes, `finish` |
| [S6](S6-single-graph-skills.md) | Control: the agent only gathers. A fixed graph proposes, checks, reviews and writes | One agent, skills, the same read tools |

## Variants

| Variant ID | Agent strategy |
|---|---|
| `s2-a` | react |
| `s2-b` | plan_execute |

## Architecture: who decides what

The same as S1:

| The agent decides | Code decides |
|---|---|
| Which records and policy files to read | Whether a write is allowed (rules on the recorded facts) |
| Whether and what to write (`create_return_request`, `request_refund`, `cancel_order`) | Whether the cited files or clauses were read this turn |
| When to ask for review (`request_review`) | Whether the write needs a review (rules' route, made stricter by the agent's `route` or `policy_exception`) |
| How the turn ends (`finish`: resolved, awaiting_customer, awaiting_human, with a note) | The review verdict (a separate approver call) |
| | The reply, its checks and the ticket save |

## Flow: a turn

See the **Flow** tab. The top row of the tab shows the stores behind the tools: Postgres for reads, and the policy path as two separate calls, `policy(open)` → the skill's SKILL.md, then `policy(read)` → a clause file it lists. `rules.json` feeds the write gate, and executed writes and the ticket land in Postgres.

The same steps as S1:

1. The guard screens the new customer message. A flagged message gets the hold reply, and no agent runs.
2. The agent works by its strategy with the seven read tools, `policy`, the three named write tools,
   `request_review` and `finish`.
3. A named write tool goes through the write gate and returns `executed`, `blocked`, `review_required`
   (call `request_review`, then write again) or `awaiting_human` (the turn ends).
4. The agent ends with `finish`. If it stops without calling it, or its strategy gives up, the turn goes to a person.
5. The shared reply writer, reply check and ticket save run.

## Data flow: what each LLM call sees

See the **Data flow & access** tab.

| LLM role | Sees | Does not see |
|---|---|---|
| Guard | The new customer message only | History, records, policy |
| Agent | The brief (message, history, signed-in account), the skill catalog up front (each skill's description and file names), its own tool calls and results | Nothing is hidden from it: it holds every tool |
| Approver (write) | The proposed write, its evidence, the rule check and `review_reason` | The agent's reasoning |
| Reply writer | The status, the agent's `finish` note, all tool evidence, problems code found | |
| Approver (reply) | The draft reply, the ticket and the evidence | |

## Data access

| Component | Tools | Reads | Writes |
|---|---|---|---|
| Agent | the seven read tools | orders, order_items, products, shipments, shipment_items, shipment_events, payments, returns, return_items, refunds | none directly |
| Agent | `policy` (`list`, `open`, `read`, `check`) | `data/policy/skills/<skill>/SKILL.md` and the clause files it lists; rules.json for `check` | none |
| Agent | `create_return_request`, `request_refund`, `cancel_order`, `request_review` | through the write gate: orders, order_items, shipments, payments, returns/refunds, rules.json | returns + return_items, refunds (pending), orders/order_items/shipments (cancel), backend_writes |
| Ticket save (code) | none | | support_tickets |

All reads and writes are scoped to the signed-in account.

## How the variants differ

**`s2-a`: react.** It behaves as `s1-a`: one call per step, an approver-route write as write →
`request_review` → write, and a nudge if it stops without `finish`. Each policy file costs two steps
(`open`, then `read`), so a typical refund takes about two more steps than in `s1-a`.

**`s2-b`: plan_execute.** It behaves as `s1-b`: a first plan for facts and policy, then a completion check
that plans the write and `finish`. A `blocked` or `review_required` result uses one of the 2 replans. With
the skill catalog up front, the first plan can name the files directly, for example `open
returns-and-refunds`, then `read refunds.md`. It cannot cite a clause ID heading it has not read yet, so the
write still waits for the completion check.

## Worth knowing

- **Citations.** A write cites the file it read (`returns-and-refunds/refunds.md`, or `refunds.md`) or a
  clause ID heading inside it. A bare `SKILL.md` is not enough, because every skill has one.
- **Tool budget.** Each `open` and `read` counts against the 40 tool calls, so S2 spends more of it on policy than S1.
- **Common errors.** Passing `file` to `open`, or using `read` on SKILL.md, is refused with the correction in
  the message. A second invalid call in the turn stops it.
- The single agent's prompt does not tell it to refund a cancelled order's captured balance. Only `s2-b`'s
  completion check prompts for it.
