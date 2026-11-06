# S1 / S2: single agent, agentic control

Code: `setups/single_agentic.py`. Diagram: [S1-S2-single-agentic.drawio](S1-S2-single-agentic.drawio).
S1 uses RAG, S2 skills. Variants: `a` = react, `b` = plan_execute.

## Who decides what

One agent holds every tool and decides what to read, when to write, when to ask for review and when it
is finished. Code only enforces the budget and the shared gates.

| The agent decides | Code decides |
|---|---|
| Which records and policy to read | Whether a write is allowed (rules on the recorded facts) |
| Whether and what to write (`create_return_request`, `request_refund`, `cancel_order`) | Whether the citations were retrieved this turn |
| When to ask for review (`request_review`) | Whether the write needs a review (rules' route, made stricter by the agent's `route` or `policy_exception`) |
| How the turn ends (`finish`: resolved, awaiting_customer, awaiting_human, with a note) | The review verdict (a separate approver call) |
| | The reply, its checks and the ticket save |

## A turn

1. The guard screens the new customer message.
2. The agent works by its strategy with the registered order lookup tools, `policy`, the three named write
   tools, `request_review` and `finish`.
3. A named write tool returns one of:
   - `executed`;
   - `blocked` (the rules refused it, or the order's returns and refunds were not looked up yet, which are
     listed once);
   - `review_required` (approver route: call `request_review` with the same write, then write again);
   - `awaiting_human` (the turn ends).
4. The agent ends with `finish`. If it stops without calling it, it is nudged once, then handed off.
5. The shared reply, reply check and ticket save run.

## Worth knowing

- A write requiring review never runs unless `request_review` approved that exact write (the same
  operation, order, amount and items).
- A repeated write returns its first result; nothing is written twice.
- After a cancellation, the prompt asks the agent to refund the captured balance with a second write.
