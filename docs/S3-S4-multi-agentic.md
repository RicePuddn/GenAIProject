# S3 / S4: supervisor and specialists, agentic control

Code: `setups/multi_agentic.py`, shared pieces in `setups/multi_common.py`. Diagram:
[S3-S4-multi-agentic.drawio](S3-S4-multi-agentic.drawio). S3 uses RAG, S4 skills.

| Variant | Supervisor | Specialists |
|---|---|---|
| a | react | react |
| b | plan_execute | plan_execute |
| c | react | plan_execute |
| d | plan_execute | react |

## Who decides what

- **The supervisor** decides which specialist handles each part, and how the ticket ends. It has
  `ask_specialist(specialist, request)` and `finish`. With react it asks one part at a time, seeing each
  report. With plan_execute it plans every part first and replans on an error or for a missing part,
  such as the refund a cancellation is owed. It never calls data tools.
- **Each specialist** acts on its part as S1's agent does, then finishes with `finish`. Its report
  (outcome, note, writes made) goes back to the supervisor.
  - orders_shipping: reads, shipments, tracking and items; scoped policy; writes; `finish`.
  - returns_refunds: reads, items, payments, returns and shipments; scoped policy; writes; `finish`.

Each specialist's `policy` tool keeps to its own part of the policy by default (RAG: its sections; skills:
its skill folder), and a call with `scope: all` reaches the rest. Write operations are not limited per
specialist; the prompts say which are whose.

## A turn

1. The guard screens the message and history.
2. The supervisor asks specialists for parts. Each specialist runs with the brief plus `your_task`.
3. A specialist's writes go through the same write gate as S1/S2, with `request_review` for the approver route.
4. The supervisor's `finish` ends the turn. If a specialist's write is handed to a person, the whole turn
   ends there.
5. The shared reply, reply check and ticket save run.

## Worth knowing

- All agents share the ticket's 30 agent steps, including the supervisor's own decisions.
- A specialist that stops without `finish` reports an error, and the supervisor decides what to do next.
