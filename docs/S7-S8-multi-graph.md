# S7 / S8: supervisor and specialists, graph control

Code: `setups/multi_graph.py`, part stages in `shared/flow.py`, shared pieces in `setups/multi_common.py`.
Diagram: [S7-S8-multi-graph.drawio](S7-S8-multi-graph.drawio). S7 uses RAG, S8 skills. The variants are
the same as S3/S4 (a: react/react, b: plan_execute/plan_execute, c: react/plan_execute, d:
plan_execute/react, supervisor first).

## Who decides what

- **The supervisor only delegates.** Its one tool is `ask_specialist(specialist, request)`. With react it
  asks one part at a time, seeing each part's outcome. With plan_execute it plans every part first, and
  replans on an error or for a missing part. It does not classify the message, redirect it, write or end
  the ticket: there is no triage.
- **Each part runs the same fixed stages as S5/S6**, with the specialist in the gather stage: gather (its
  reads and scoped policy) → propose → rule check → review (approver route) → write. Refusals go back to
  gather with the reason.
- **Code decides the ticket's outcome:** the strictest of the parts (awaiting_human, then
  awaiting_customer, then resolved).

## A turn

1. The guard screens the message and history.
2. The supervisor asks for parts. Each part's report (outcome, note, writes) goes back to it.
3. When the supervisor stops asking, the strictest outcome decides the status.
4. A handoff in any part (a proposal of awaiting_human, or the write gate) ends the whole turn.
5. The shared reply, reply check and ticket save run.

## Worth knowing

- If the supervisor asks no specialist (for example, an off-topic message), the ticket is resolved and the
  reply says what support covers.
- Specialists' policy scope and tool subsets are the same as in S3/S4.
