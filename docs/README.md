# Setup walkthroughs

One page and one draw.io diagram per code path. Each pair of setups runs the same code; only `policy.mode`
(RAG or skills) differs. Open a `.drawio` file in draw.io (app.diagrams.net) or the VS Code Draw.io extension.

| Setups | Page | Diagram | Owners |
|---|---|---|---|
| S1, S2 | [single agent, agentic](S1-S2-single-agentic.md) | [S1-S2-single-agentic.drawio](S1-S2-single-agentic.drawio) | FP, Ernest |
| S3, S4 | [multi-agent, agentic](S3-S4-multi-agentic.md) | [S3-S4-multi-agentic.drawio](S3-S4-multi-agentic.drawio) | FT, Lynn |
| S5, S6 | [single agent, graph](S5-S6-single-graph.md) | [S5-S6-single-graph.drawio](S5-S6-single-graph.drawio) | FP, Ernest |
| S7, S8 | [multi-agent, graph](S7-S8-multi-graph.md) | [S7-S8-multi-graph.drawio](S7-S8-multi-graph.drawio) | FT, Lynn |

## What every setup shares

Every turn runs `shared/pipeline.py::run_turn`: **guard → setup body → reply writer → wording checks →
reply check → ticket save**. Only the body differs between setups. The rest is the same code everywhere:

- **Guard.** Prompt Guard screens the message and every history turn. A flagged turn gets the fixed hold
  reply, and no agent runs.
- **Write gate**, the only write path, in this order:
  1. The rules on the recorded facts.
  2. Each cited clause must be one a policy search or read returned this turn.
  3. Returns or refunds on the order that no lookup has shown this turn are listed once before the write can go on.
  4. A write may ask for a stricter route (`route: approver | human`), and one that sets `policy_exception` gets the approver.
  5. On the approver route, an independent review.
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
- **Budget.** One per ticket, shared by every agent in it: 40 tool calls, 30 agent steps, US$0.05. Hitting
  it ends the turn with the hold reply.

**Strategies.** One loop each, used by every agent:

- `react` makes one tool call per turn.
- `plan_execute` plans up front, runs the plan, replans on an error or a block (up to 2 times), and after
  a plan runs in full it checks once more for a missing step (up to 3 times). The supervisor plans the
  parts of the message; a working agent plans its own tool calls.
