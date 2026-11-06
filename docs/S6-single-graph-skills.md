# S6: single agent · graph control · skills policy

Code: `setups/single_graph.py` (shared with S5), stages in `shared/flow.py`. Diagram:
[S6-single-graph-skills.drawio](S6-single-graph-skills.drawio) (tabs: **Flow**, **Data flow & access**). Owner: Ernest.
Shared pipeline, gates and strategies: [README](README.md).

## At a glance

| | S6 |
|---|---|
| Agents | One agent, used only in the gather stage and for the proposal |
| Control | Graph: a fixed LangGraph sequence decided in code, gather → propose → rule check → review → write |
| Policy | Skills: `open` a skill's SKILL.md, then `read` its clause files under `data/policy/skills/` |
| Who writes | Code (the graph's write stage), from the agent's proposal |
| Who ends the ticket | Code: the proposal's outcome, unless something hands it off |
| Variants | `s6-a` (react gather), `s6-b` (plan_execute gather) |

## How S6 differs from its neighbours

| Compared with | What changes | What stays the same |
|---|---|---|
| [S5](S5-single-graph-rag.md) | Policy: skills files instead of RAG search | Everything else: same code, same variants |
| [S8](S8-multi-graph-skills.md) | Agents: a supervisor asks specialists, each kept to its own skill folder, and each part runs these stages separately | Graph stages, skills, no agent writes |
| [S2](S2-single-agentic-skills.md) | Control: the agent holds the write tools, asks for review and ends the ticket itself | One agent, skills, the same read tools |

## Variants

| Variant ID | Gather-stage strategy |
|---|---|
| `s6-a` | react |
| `s6-b` | plan_execute |

## Architecture: who decides what

The same as S5. The agent decides what to read and what to propose. Code decides when gathering ends,
whether each write is allowed, reviewed and executed, sending refusals back to gather, and ending the turn on
a handoff. The agent has no write tools, no `request_review` and no `finish`.

## Flow: a turn

See the **Flow** tab. The top row of the tab shows the stores behind the tools: Postgres for reads, and the policy path as two separate calls, `policy(open)` → the skill's SKILL.md, then `policy(read)` → a clause file it lists. `rules.json` feeds the write gate, and executed writes and the ticket land in Postgres.

The same stages as S5:

1. The guard screens the new customer message.
2. **Gather** with the seven read tools and `policy` (`list`, `open`, `read`, `check`), until the agent stops asking for tools.
3. **Propose**: outcome, note, writes, review_reason.
4. For each write: **rule check** → **review** (approver route only) → **write** (`backend_write`).
5. Refusals go back to gather with the reason. A handoff ends the turn.
6. Otherwise the proposal's outcome is the status. The shared reply writer, reply check and ticket save run.

## Data flow: what each LLM call sees

| LLM role | Sees | Does not see |
|---|---|---|
| Guard | The new customer message only | |
| Agent (gather) | The brief, the skill catalog up front (descriptions and file names), its own calls and results. After a refusal, also its earlier evidence and the `feedback` | Write results, until the graph refuses one |
| Agent (propose) | The brief, all its own evidence (including the full text of each file read), any `feedback` | |
| Approver (write) | The proposed write, its evidence, the rule check and `review_reason` | |
| Reply writer | The status, the proposal's note, all evidence, problems code found | |

## Data access

| Component | Tools | Reads | Writes |
|---|---|---|---|
| Agent (gather) | the seven read tools | orders, order_items, products, shipments, shipment_items, shipment_events, payments, returns, return_items, refunds | none |
| Agent (gather) | `policy` (`list`, `open`, `read`, `check`) | `data/policy/skills/<skill>/SKILL.md` and its clause files; rules.json for `check` | none |
| Graph write stages (code) | `backend_write` (internal) | orders, order_items, shipments, payments, returns/refunds, rules.json | returns + return_items, refunds (pending), orders/order_items/shipments (cancel), backend_writes |
| Ticket save (code) | none | | support_tickets |

All reads and writes are scoped to the signed-in account.

## How the variants differ

**`s6-a`: react gather.** As `s5-a`, but each policy file is two steps (`open`, then `read`). Gathering ends
when the agent stops calling tools. 3 identical errors in a row end the turn with a person.

**`s6-b`: plan_execute gather.** As `s5-b`. Because the catalog lists file names up front, the first plan can
name exact files (`open returns-and-refunds`, `read eligibility.md`). It ends on an empty plan. No working plan
after 2 replans, or still planning after 3 checks, ends the turn with a person.

**Both:** a refusal restarts gather with its earlier evidence and the `feedback`.

## Worth knowing

- **Citations.** The proposal cites a file (`skill/file` or the file name) or a clause ID heading in it. A
  bare `SKILL.md` is not citable. A file the agent never read is refused at the rule check and sends it back to gather.
- **Tool budget.** Two calls per policy file, each counting against the 40 tool calls, shared with graph write attempts.
- A proposal of `awaiting_human` ends the turn with nothing written.
