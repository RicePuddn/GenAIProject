# S4: supervisor + specialists · agentic control · skills policy

Code: `setups/multi_agentic.py` (shared with S3), supervisor and specialists in `setups/multi_common.py`.
Diagram: [S4-multi-agentic-skills.drawio](S4-multi-agentic-skills.drawio) (tabs: **Flow**, **Data flow & access**).
Owner: Lynn. Shared pipeline, gates and strategies: [README](README.md).

## At a glance

| | S4 |
|---|---|
| Agents | A supervisor and two specialists, `orders_shipping` and `returns_refunds` |
| Control | Agentic: the supervisor decides who handles each part. Each specialist decides its own reads, writes and reviews |
| Policy | Skills: each specialist's `policy` tool keeps by default to its own skill folder (`scope: own`) |
| Who writes | The specialists, through the named write tools |
| Who ends the ticket | The supervisor, with `finish`, but never milder than the strictest specialist report |
| Variants | `s4-a` … `s4-d` (supervisor × specialist strategy) |

## How S4 differs from its neighbours

| Compared with | What changes | What stays the same |
|---|---|---|
| [S3](S3-multi-agentic-rag.md) | Policy: skills folders instead of RAG sections | Everything else: same code, same variants |
| [S2](S2-single-agentic-skills.md) | Agents: one agent with every tool and every skill instead of a supervisor and two scoped specialists | Agentic control, skills, named writes, `finish` |
| [S8](S8-multi-graph-skills.md) | Control: specialists only gather inside fixed graph stages, and the supervisor has no `finish` | Supervisor + specialists, skills, same tool subsets and scoping |

## Variants

| Variant ID | Supervisor | Specialists |
|---|---|---|
| `s4-a` | react | react |
| `s4-b` | plan_execute | plan_execute |
| `s4-c` | react | plan_execute |
| `s4-d` | plan_execute | react |

## Architecture: who decides what

The same as S3:

- **The supervisor** decides which specialist handles each part, and how the ticket ends (`ask_specialist`,
  `finish`). It never sees records or policy.
- **Each specialist** acts on its part as S2's agent does, then reports back with `finish`.
- **Code sets a floor on the outcome**: never milder than the strictest report.
- **Code decides every write**, through the shared write gate.

## Flow: a turn

See the **Flow** tab. The top row of the tab shows the stores behind the tools: Postgres for reads, and the policy path as two separate calls, `policy(open)` → the skill's SKILL.md, then `policy(read)` → a clause file it lists. `rules.json` feeds the write gate, and executed writes and the ticket land in Postgres. A specialist keeps by default to its own skill folder.

The same steps as S3:

1. The guard screens the new customer message.
2. The supervisor asks a specialist for each part. Each ask runs the specialist fresh, with the brief plus `your_task`.
3. A specialist's writes go through the write gate. A write handed to a person ends the whole turn.
4. The specialist ends with `finish`, and its report goes back to the supervisor.
5. The supervisor's `finish` ends the turn, raised to the strictest part's outcome if needed.
6. The shared reply writer, reply check and ticket save run.

## Data flow: what each LLM call sees

| LLM role | Sees | Does not see |
|---|---|---|
| Guard | The new customer message only | |
| Supervisor | The brief, and each report: specialist, request, outcome, note, writes | Any record, tool result or policy text |
| Specialist | The brief plus `your_task`, **its own** skills in the catalog up front, its own tool calls and results in this ask | The other folder's catalog (unless it lists with `scope: all`), the other specialist's work, its own earlier asks |
| Approver (write) | The proposed write, its evidence, the rule check and `review_reason` | |
| Reply writer | The final status and note, all tool evidence from every specialist, problems code found | |

## Data access

| Component | Read tools | Policy | Writes |
|---|---|---|---|
| Supervisor | none | none | none |
| orders_shipping | `get_order`, `get_customer_orders`, `get_shipments`, `get_tracking_events`, `get_order_items` | skill folder `orders-and-shipping`; `scope: all` for the rest | named writes + `request_review` (by prompt: `cancel_order`) |
| returns_refunds | `get_order`, `get_customer_orders`, `get_order_items`, `get_payments`, `get_returns_and_refunds`, `get_shipments` | skill folder `returns-and-refunds`; `scope: all` for the rest | named writes + `request_review` (by prompt: `create_return_request`, `request_refund`) |

**Skills scope.** With `scope: own`, `list`, `open` and `read` keep to the specialist's folder. Opening the
other folder is refused with "use scope all". A guessed skill name gets back only the specialist's own skill
names. This is stricter than S3's RAG scope: there, the specialist still sees every section name and the
search just filters hits.

## How the variants differ

The same mechanics as S3. See [S3: How the variants differ](S3-multi-agentic-rag.md#how-the-variants-differ).

- **Supervisor react (`s4-a`, `s4-c`):** one part at a time, each chosen after the last report. A nudge if it stops without `finish`.
- **Supervisor plan_execute (`s4-b`, `s4-d`):** all parts planned up front. A completion check adds open parts,
  such as the refund a cancelled order is owed.
- **Specialists react (`s4-a`, `s4-d`):** as `s2-a`. Each policy file is two steps (`open`, `read`).
- **Specialists plan_execute (`s4-b`, `s4-c`):** as `s2-b`. Their catalog lists only their own skills, so a
  first plan names files from their own folder.

## Worth knowing

- A failed specialist reports `awaiting_human`, and the outcome floor then ends the ticket `awaiting_human`,
  even if a retry succeeds (as in S3).
- Each ask starts fresh. Re-asking repeats the specialist's `open` and `read` calls unless the supervisor's
  `request` says what is known, and each one counts against the 40 tool calls.
- A write cites the file read (`skill/file`, or the file name) or a clause ID heading inside it.
