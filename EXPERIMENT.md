# Experiment design (draft)

What we are comparing, what each setup is, and what must stay the same across setups.
If code, configs or other docs disagree with this file, this file wins and the other one gets fixed.

## Hypotheses

- **H1 (S1–S4):** letting agents make their own tool calls is more flexible and may be faster.
- **H2 (S5–S8):** a deterministic, code-controlled flow succeeds more often and gives more consistent results.

## The eight setups

Three factors, two levels each. Every variant is one config file in `configs/setups/`.

| Setup | Agents | Control | Policy | Owner | Variants |
|---|---|---|---|---|---|
| S1 | Single | Agentic | RAG   | FP     | 2 |
| S2 | Single | Agentic | SKILL | Ernest | 2 |
| S3 | Multi  | Agentic | RAG   | FT     | 4 |
| S4 | Multi  | Agentic | SKILL | Lynn   | 4 |
| S5 | Single | Graph   | RAG   | FP     | 2 |
| S6 | Single | Graph   | SKILL | Ernest | 2 |
| S7 | Multi  | Graph   | RAG   | FT     | 4 |
| S8 | Multi  | Graph   | SKILL | Lynn   | 4 |

**24 variants in total.** Each pair (S1/S2, S3/S4, S5/S6, S7/S8) must run the **same code**, with only `policy.mode` changed.
The pair's two owners share that code. They do not keep separate copies.

## Definitions

**Agentic control (S1–S4).** The agent holds every tool, including `backend_write`, and decides for itself
what to read, when to write, when to ask for review and when it is finished.
Code only enforces budgets and the shared gates below.

**Graph control (S5–S8).** A fixed LangGraph sequence of stages, decided in code:
gather facts → propose an action (structured output) → rule check → review if required → write → reply.
The agent works only inside the gather stage, with read and policy tools. It cannot write, start a review or end
the ticket itself; it hands over a proposal, and the graph decides what happens next.
S7/S8 have **no triage step**: the supervisor does not classify or redirect the message.

**Single agent.** One agent does all the work.

**Multi-agent.** A supervisor plus two specialists, `orders_shipping` and `returns_refunds`, each with its own subset of the tools.
The supervisor delegates parts of the ticket. Only the specialists call data tools.

**RAG.** The `policy` tool searches the Pinecone index of the policy PDFs (`policy.rag` settings).
**SKILL.** The `policy` tool opens `SKILL.md` files under `policy/skills/` and reads the clause files they point to.
Both modes cover the same policy text.

**Strategies.**

- `react`: one step at a time, each chosen after seeing the previous result.
- `plan_execute`: plan all steps up front, run them, then replan only on an error or for a missing step.

For a multi-agent setup, the supervisor's strategy decides how it delegates (one part at a time, or all parts planned first).
The specialists' strategy decides how they call tools.

## Variants

The same variants apply to the agentic setup and its graph counterpart.

| Variant | S1, S2, S5, S6: agent | S3, S4, S7, S8: supervisor | S3, S4, S7, S8: specialists |
|---|---|---|---|
| a | react        | react        | react |
| b | plan_execute | plan_execute | plan_execute |
| c | –            | react        | plan_execute |
| d | –            | plan_execute | react |

Config files are named by setup and variant: `s1-a.yaml`, `s3-c.yaml`, and so on.

## Held constant across all setups

Any difference between setups on the items below is a bug. It is not a design choice.

| Item | Value / where |
|---|---|
| Model (all roles except the guard) | `openai/gpt-oss-120b` on Groq, temperature 0, `reasoning_effort: medium` for every role |
| Prompt attack guard | `llama-prompt-guard-2-86m`, threshold 0.5, screens every turn including history |
| Data tools | One tool registry. Multi-agent setups give each specialist a subset; they never get different implementations |
| Business rules | One rules engine, reading `src/support_ops/policy/rules/rules.json` |
| Write path | One write gate (`gateway/write_gate.py`): rule check → cited clauses must be ones retrieved in this ticket → independent approver review when required. Agentic setups reach it through the `backend_write` tool; graph setups call it as a node |
| Write review and reply check | Same approver prompts and limits (`review.max_rejections: 1`, `review.max_reply_drafts: 2`) |
| Reply | Same reply prompt and the same wording checks (no "refunded" claims, no internal IDs) |
| Ticket save | Saved in code at the end of every turn, the same way in every setup |
| Budgets | 40 tool calls, 12 agent steps and US$0.05 per ticket |
| Clock | Fixed at `run.clock_start` (2026-10-01 09:00 +08:00) |

Metrics and the evaluation run are deferred; they will be added here later.

## Where the code stands (2026-11-01)

These gaps block a fair comparison. Fix them roughly in this order.

1. **The setup labels in the code are wrong.** `systems/multi_agent.py` and `docs/multi-agent.md` call the fixed
   workflow "S3/S4". In this plan it is S7/S8. `systems/supervisor_agent.py` is the agentic S3/S4.
2. **The shared gates are copied rather than shared.** Guard, reply check and ticket save are re-implemented in each
   workflow. Wrap every setup as `guard → setup body → reply check → save ticket`, so setups implement only the body.
3. **Two write paths.** The fixed multi-agent workflow writes through `route` → approval tokens (`gateway/approvals.py`,
   `gateway/writes.py`). Every other setup writes through `gateway/write_gate.py`. Move S7/S8 onto the write gate.
4. **Triage still exists in the fixed multi-agent workflow** (the supervisor classifies and can redirect). Remove it for S7/S8.
5. **S5/S6 are not built.**
6. **The agentic multi-agent class inherits from the fixed one.** `SupervisorAgentWorkflow` (S3/S4) subclasses
   `MultiAgentWorkflow` (S7/S8) and replaces its graph. Separate them, sharing code through the common pieces instead.
7. **Two APIs and two demo apps** with different request schemas. Merge them into one, with the setup and variant chosen by config.
8. **No per-variant config files.** Only `multi_agent_rag.yaml` and `multi_agent_skills.yaml` exist. Add the 24 files under `configs/setups/`.

## Open questions for the team

- Is the ownership above right? The code was split by agentic vs fixed workflow (Lynn/FT); this table splits by policy mode.
