# System walkthroughs

One page and one draw.io diagram per system. Each diagram has two tabs: **Flow** (the control flow of a turn,
with the stores behind it: Postgres, `rules.json`, and the policy store, drawn as the RAG retriever and Pinecone
index or as the skills files, so each RAG/skills pair looks different) and **Data flow & access** (what each LLM call sees, and which tools and data each layer touches). Open a
`.drawio` file in draw.io (app.diagrams.net) or the VS Code Draw.io extension.

Each pair (S1/S2, S3/S4, S5/S6, S7/S8) runs the same code; only `policy.mode` differs. Each page opens with
the three neighbours that differ from it in exactly one way.

| System | Page | Diagram | Owner |
|---|---|---|---|
| S1 | [single · agentic · RAG](S1-single-agentic-rag.md) | [S1-single-agentic-rag.drawio](S1-single-agentic-rag.drawio) | FP |
| S2 | [single · agentic · skills](S2-single-agentic-skills.md) | [S2-single-agentic-skills.drawio](S2-single-agentic-skills.drawio) | Ernest |
| S3 | [multi · agentic · RAG](S3-multi-agentic-rag.md) | [S3-multi-agentic-rag.drawio](S3-multi-agentic-rag.drawio) | FT |
| S4 | [multi · agentic · skills](S4-multi-agentic-skills.md) | [S4-multi-agentic-skills.drawio](S4-multi-agentic-skills.drawio) | Lynn |
| S5 | [single · graph · RAG](S5-single-graph-rag.md) | [S5-single-graph-rag.drawio](S5-single-graph-rag.drawio) | FP |
| S6 | [single · graph · skills](S6-single-graph-skills.md) | [S6-single-graph-skills.drawio](S6-single-graph-skills.drawio) | Ernest |
| S7 | [multi · graph · RAG](S7-multi-graph-rag.md) | [S7-multi-graph-rag.drawio](S7-multi-graph-rag.drawio) | FT |
| S8 | [multi · graph · skills](S8-multi-graph-skills.md) | [S8-multi-graph-skills.drawio](S8-multi-graph-skills.drawio) | Lynn |

## The eight systems side by side

| | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 |
|---|---|---|---|---|---|---|---|---|
| Agents | single | single | multi | multi | single | single | multi | multi |
| Control | agentic | agentic | agentic | agentic | graph | graph | graph | graph |
| Policy | RAG | skills | RAG | skills | RAG | skills | RAG | skills |
| Who holds the read tools | agent (all 7) | agent (all 7) | specialists (subsets) | specialists (subsets) | agent (all 7) | agent (all 7) | specialists (subsets) | specialists (subsets) |
| Policy scope | all | all | own section | own skill folder | all | all | own section | own skill folder |
| Who writes | agent | agent | specialists | specialists | graph (code) | graph (code) | graph (code), per part | graph (code), per part |
| Who asks for review | agent (`request_review`) | agent | specialists | specialists | graph | graph | graph | graph |
| Who ends the ticket | agent (`finish`) | agent | supervisor (`finish`), floored at the strictest part | supervisor, floored | proposal's outcome | proposal's outcome | strictest part (code) | strictest part (code) |
| What the strategy drives | the agent | the agent | supervisor + specialists | supervisor + specialists | gather only | gather only | supervisor + gather | supervisor + gather |
| Variants | 2 | 2 | 4 | 4 | 2 | 2 | 4 | 4 |

## The 24 variants

A variant combines four choices: agent count (single or multi), control style (agentic or graph), policy mode
(RAG or skills), and each role's strategy (react or plan_execute). Single-agent setups have one role, so they
have 2 variants. Multi-agent setups pick a strategy for the supervisor and for the specialists separately, so
they have 4. That gives 4 × 2 + 4 × 4 = **24**. Each ID resolves from the one matrix in
[configs/setups.yaml](../configs/setups.yaml); no variant has its own file or code.

| Agents | Control | RAG | Skills | Variant IDs | Code |
|---|---|---|---|---|---|
| single | agentic | [S1](S1-single-agentic-rag.md) | [S2](S2-single-agentic-skills.md) | `s1-a` `s1-b` · `s2-a` `s2-b` | `setups/single_agentic.py` |
| multi | agentic | [S3](S3-multi-agentic-rag.md) | [S4](S4-multi-agentic-skills.md) | `s3-a` … `s3-d` · `s4-a` … `s4-d` | `setups/multi_agentic.py` |
| single | graph | [S5](S5-single-graph-rag.md) | [S6](S6-single-graph-skills.md) | `s5-a` `s5-b` · `s6-a` `s6-b` | `setups/single_graph.py` |
| multi | graph | [S7](S7-multi-graph-rag.md) | [S8](S8-multi-graph-skills.md) | `s7-a` … `s7-d` · `s8-a` … `s8-d` | `setups/multi_graph.py` |

| Variant | Single agent (S1, S2, S5, S6) | Supervisor (S3, S4, S7, S8) | Specialists (S3, S4, S7, S8) |
|---|---|---|---|
| a | react | react | react |
| b | plan_execute | plan_execute | plan_execute |
| c | – | react | plan_execute |
| d | – | plan_execute | react |

Pick one per request with `"setup": "s3-c"`, or set `run.setup` in `configs/base.yaml`. `GET /info` lists them all.

## What every setup shares

Every turn runs `shared/pipeline.py::run_turn`: **guard → setup body → reply writer → wording checks →
reply check → ticket save**. Only the body differs between setups. The rest is the same code everywhere:

- **Guard.** The guard model screens the new customer message. A flagged turn gets the fixed hold
  reply, and no agent runs.
- **Write gate**, the only write path, in this order:
  1. A write this ticket already made is replayed, not repeated.
  2. The rules on the recorded facts.
  3. Each cited clause must be one a policy search or read returned this turn.
  4. Returns or refunds on the order that no lookup has shown this turn are listed once before the write can go on.
  5. A write may ask for a stricter route (`route: approver | human`), and one that sets `policy_exception` gets the approver.
  6. On the approver route, an independent review.
- **Rejections.** One review rejection can be fixed; the next one, or an escalation, goes to a person.
- **Handoffs.** When the gate hands a write to a person, the whole turn ends `awaiting_human`.
- **Reply.** The reply writer drafts from the ticket: its status, the agent's note, the tool evidence, and
  problems code found in what the agents looked up (an overdue shipment, a return the store hasn't
  processed). Code rejects paid-refund claims and internal IDs. Then the approver checks the reply. If no
  draft passes, the customer gets the hold reply.
- **Ticket save.** The ticket is saved in code at the end of every turn.
- **Failures.** A tool or write that fails for a reason the model can't fix is retried once. If the LLM
  provider fails, the ticket is saved `in_progress` and the customer is asked to send the message again. Writes
  already made replay rather than repeat. A tool call that keeps failing does the same to a turn that would
  otherwise end `resolved`, so an agent cannot resolve a ticket over work that never ran.
- **Invalid tool calls.** An agent's first call with invalid arguments gets back the fields to correct.
  Its second one in the turn stops the turn and hands it to a person.
- **Budget.** One per ticket, shared by every agent in it: 40 tool calls, 30 agent steps, US$0.05. Hitting
  it ends the turn with the hold reply.

## Strategies

One loop each (`shared/strategies/`), used by every agent: the single agent, the supervisor and the
specialists. Only the toolset changes what the loop does. An agent with a `finish` tool ends when it calls
it (S1–S4's working agents and S3/S4's supervisor). An agent without one ends when it stops calling tools
(S5–S8's gather stage and S7/S8's supervisor).

**`react`** (`react.py`): one tool call per LLM step, each chosen after seeing the previous result.

- If the model returns several tool calls, only the first runs. The rest are dropped and logged as `tool_calls_deferred`.
- An agent with `finish` that replies without a tool call is nudged once. The second time, it is handed off.
- The same tool failing with the same error 3 times in a row ends the run (`repeated_error`).
- Each step costs one agent step from the ticket's 30.

**`plan_execute`** (`plan_execute.py`): plan the calls up front as JSON, run them, and replan only when needed.

- The whole plan is checked before any of it runs: known tool names, valid arguments, `finish` only as the
  last call, and enough tool budget left for its data calls.
- A plan may not use a value it has not seen yet, such as a clause ID to cite or an amount from the payments.
  It stops after the calls that fetch it.
- **Replan** (up to `limits.max_replans` = 2): a call that errors, or a write that comes back `blocked` or
  `review_required`, stops the plan. The rest of the plan does not run, and the agent replans with
  `previous_error`. One more failure after that ends the run.
- **Completion check** (up to `limits.max_plan_checks` = 3): after a plan runs in full, the agent is asked
  once more, with the results, for anything still missing. This is where the write that needed a clause ID or
  an amount gets planned, and where an agent with `finish` finishes. A plan that still is not done after 3
  checks ends the run.
- A plan that is not JSON at all is asked for again once (`retries.invalid_json`), then the run ends.
- The supervisor plans the parts of the message (`plan_parts`, one `ask_specialist` per part). A working agent
  plans its own tool calls (`plan_calls`).

When a run ends without finishing, the setup decides what happens. Every setup hands the turn to a person.

## Policy modes

Each pair differs only in the provider behind the one `policy` tool (`shared/policy/`). The write gate, the
rules and the delivery check are the same in both.

| | RAG (S1, S3, S5, S7) | Skills (S2, S4, S6, S8) |
|---|---|---|
| Tool actions | `search` (query), `check` | `list`, `open` (a skill's SKILL.md), `read` (a clause file), `check` |
| What comes back | The top 4 clauses (hybrid dense + BM25, reranked) and the sections they reference | The whole file |
| What a write may cite | The returned clause IDs | The file read (`skill/file`, or the file name), or a clause ID heading inside it |
| Up-front context | The policy section names | The skill catalog: each skill's description and files |
| Typical cost | One call per policy question | At least two calls (`open`, then `read`), each counting against the 40 tool calls |
| Specialist scope (`scope: own`) | Search hits are filtered to its sections and the clauses they reference. It still sees every section name | `list`, `open` and `read` keep to its skill folder, and its catalog lists only its own skills |

`scope: all` lifts a specialist's scope in either mode. `check` (operation `delivery`) runs the rules engine
on each parcel. Its `rule_sources` cannot be cited.
