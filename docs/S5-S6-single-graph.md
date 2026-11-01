# S5 / S6: single agent, graph control

Code: `setups/single_graph.py`, stages in `shared/flow.py`. Diagram:
[S5-S6-single-graph.drawio](S5-S6-single-graph.drawio). S5 uses RAG, S6 skills. Variants: `a` = react,
`b` = plan_execute (for the gather stage).

## Who decides what

A fixed LangGraph sequence decided in code. The agent works only inside the gather stage, with read and
policy tools. It cannot write, start a review or end the ticket itself: it hands over a proposal, and the
graph decides what happens next.

## A turn

1. The guard screens the new customer message.
2. **Gather.** The agent reads by its strategy (`read_record`, `get_orders_for_customer`, the order lookups,
   `policy`). React ends when it stops calling tools; plan_execute ends when its completion check plans nothing more.
3. **Propose.** One structured output: `outcome` (resolved, awaiting_customer or awaiting_human), `note`,
   `writes` (backend_write's arguments, including `route` and `policy_exception`) and `review_reason`.
4. For each write, in order:
   - **Rule check**, the write gate. A write this ticket already made is replayed, not repeated.
   - **Review**, on the approver route only.
   - **Write**.
5. Anything refused goes back to gather with the reason (`feedback`): a block, a citation that wasn't
   retrieved, unseen returns or refunds, or a fixable review rejection. A handoff, or an agent that gives
   up while gathering, ends the turn. The budget
   ends a loop of refusals.
6. The shared reply, reply check and ticket save run.

## Worth knowing

- A proposal of `awaiting_human` ends the turn with nothing written.
- Each write attempt counts and is traced as one `backend_write` tool call, as in S1/S2.
