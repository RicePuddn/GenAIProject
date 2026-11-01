# Experiment Design

## Hypotheses

- **H1 (S1–S4):** letting agents make their own tool calls is more flexible and may be faster.
- **H2 (S5–S8):** a deterministic, code-controlled flow succeeds more often and gives more consistent results.

## Setups

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
The supervisor delegates parts of the ticket. Only the specialists call data tools. Each specialist's `policy` tool
keeps by default to its own part of the policy (RAG: its sections; SKILL: its skill folder) and reaches the rest with
`scope: all`; it is the same provider as the single agent's, not a separate implementation.

**RAG.** The `policy` tool searches the Pinecone index of the policy PDFs (`policy.rag` settings).
**SKILL.** The `policy` tool opens `SKILL.md` files under `data/policy/skills/` and reads the clause files they point to.
Both modes cover the same policy text.

**Strategies.**

- `react`: one step at a time, each chosen after seeing the previous result.
- `plan_execute`: plan all steps up front, run them, then replan only on an error or for a missing step.

For a multi-agent setup, the supervisor's strategy decides how it delegates (one part at a time, or all parts planned first).
The specialists' strategy decides how they call tools. Both use the same loop; a supervisor's plan is the parts of the
message, a working agent's plan is its tool calls.

## Variants

The same variants apply to the agentic setup and its graph counterpart.

| Variant | S1, S2, S5, S6: agent | S3, S4, S7, S8: supervisor | S3, S4, S7, S8: specialists |
|---|---|---|---|
| a | react        | react        | react |
| b | plan_execute | plan_execute | plan_execute |
| c | –            | react        | plan_execute |
| d | –            | plan_execute | react |

Each variant has a setup ID such as `s1-a` or `s3-c`, resolved from the one matrix in `configs/setups.yaml` (no file per variant).

## Held constant across all setups

| Item | Value / where |
|---|---|
| Model (all roles except the guard) | `openai/gpt-oss-120b` on Groq, temperature 0, `reasoning_effort: medium` for every role |
| Prompt attack guard | `llama-prompt-guard-2-86m`, threshold 0.5, screens the new customer message once per turn |
| Data tools | One tool registry (`shared/tools.py`). Multi-agent setups give each specialist a subset; they never get different implementations (a specialist's `policy` tool is a scoped view of the same provider) |
| Business rules | One rules engine, reading `data/policy/rules.json` |
| Write path | One write gate (`shared/gates/write_gate.py`): rule check → cited clauses must be ones retrieved in this ticket → the order's returns and refunds not yet looked up are shown once → the route is the stricter of the rules' and the agent's own (`route`, `policy_exception`), never looser → independent approver review when required. Agentic setups reach it through `backend_write` and ask for the review themselves with `request_review`; graph setups call it as nodes |
| Write review and reply check | Same approver prompts and limits (`review.max_rejections: 1`, `review.max_reply_drafts: 2`) |
| Reply | Same reply prompt and the same wording checks, written from the ticket's status, the agent's note, the evidence and the problems code finds in what the agents looked up |
| Ticket save | Saved in code at the end of every turn, the same way in every setup |
| Budgets | 40 tool calls, 30 agent steps and US$0.05 per ticket, shared by every agent in it; one retry policy (`retries`). The step limit is a safety net set above what a multi-agent ticket needs, so it should rarely end a ticket; budget stops are reported per setup as their own failure type |
| Failures | A tool or write that fails for a reason the model can't fix is re-run under `retries` in every setup, graph stages included; control tools are not re-run. If the LLM provider fails, every setup ends the turn `in_progress`: the ticket is saved, the customer is asked to send the message again, and no write repeats. A ticket a person already has stays `awaiting_human`. A tool call that fails for good and never works later also stops a turn ending `resolved`: it ends `in_progress` the same way, whatever the agent concluded |
| Clock | Fixed at `run.clock_start` (2026-10-01 09:00 +08:00); only an explicit `clock_start` in an API request (the demo's "use today's date") overrides it |
| LLM cache | Off (`llm.cache_dir: null`); when on, the trial number (`run.trial`) is part of the key |

## Recommended repo structure

Organised around the setups. Everything held constant lives in `shared/`. Policy mode and strategy are config values,
so only the four combinations of agent count and control style get their own code.

```text
GenAIProject/
├── EXPERIMENT.md              study design (this file, the source of truth)
├── ARCHITECTURE.md            how the code fits together, one page
├── docs/                      one walkthrough and draw.io diagram per setup pair
├── README.md                  install and run only
├── .gitattributes             *.pdf binary
├── configs/
│   ├── base.yaml              every value held constant
│   ├── setups.yaml            S1–S8 and variants a–d; "s3-c" resolves from here
│   └── rag/                   retrieval profiles (vector, hybrid, hybrid_rerank)
├── data/
│   ├── db/                    schema.sql and seed fixtures
│   ├── policy/
│   │   ├── pdf/               source policies
│   │   ├── chunks/            section chunks (RAG)
│   │   ├── embeddings/        vectors and upload manifest (RAG)
│   │   ├── skills/            SKILL.md trees (SKILL)
│   │   └── rules.json         business rules
│   └── cases/                 test cases (later)
├── src/support_ops/
│   ├── shared/                identical for every setup
│   │   ├── llm.py  tracing.py  clock.py  config.py  budget.py  cancellation.py
│   │   ├── db/                reader.py, writer.py, accounts.py
│   │   ├── tools.py           the one tool registry
│   │   ├── rules.py           the one rules engine
│   │   ├── policy/            provider.py, rag/, skills.py, citations.py
│   │   ├── gates/             guard.py, write_gate.py, approver.py, reply.py, ticket.py, handoff.py
│   │   ├── strategies/        react.py, plan_execute.py (one loop for every agent)
│   │   ├── context.py         TurnContext: one turn's LLM, trace, clock, budget and evidence
│   │   ├── flow.py            the graph setups' stages (gather → propose → check → review → write)
│   │   └── pipeline.py        run_turn(): guard → setup body → reply check → save ticket
│   ├── prompts/               one .md per role: agent, supervisor, specialists, approver, reply writer;
│   │                          plus shared facts, control, strategies and propose text
│   ├── setups/
│   │   ├── base.py            Setup interface and TurnResult (same shape for every setup)
│   │   ├── multi_common.py    supervisor and specialist definitions shared by S3/S4/S7/S8
│   │   ├── single_agentic.py  S1/S2
│   │   ├── single_graph.py    S5/S6
│   │   ├── multi_agentic.py   S3/S4
│   │   ├── multi_graph.py     S7/S8
│   │   └── __init__.py        build_setup("s3-c")
│   └── api.py                 one API; setup and variant from config or request
├── apps/demo.py               one Streamlit page
├── scripts/                   seed_db.py, build_rag_index.py, sync_skills.py
├── notebooks/                 rag_evaluation.ipynb
├── tests/
│   ├── fakes.py               scripted LLM, fake database, fake retriever
│   ├── shared/                unit tests for gates, rules, tools, policy
│   ├── setups/                behaviour tests: message in → expected outcome and writes, for every setup
│   └── test_api.py, test_demo.py
└── results/                   run outputs (gitignored except summaries)
```

Rules for the structure:

- A setup file imports only from `shared/`, `prompts/`, `setups/base.py` and `multi_common.py`, never from another setup file.
- A setup implements only its body. It never calls the guard, the reply check or the ticket save itself.

Suggested code moves:

| Now | Moves to |
|---|---|
| `helpers/llm.py`, `tracing.py`, `clock.py`, `config.py`, `cancellation.py`, `agents/runtime.py` (Budget) | `shared/` |
| `backend/` | `shared/db/` |
| `systems/tools.py`, `agents/toolbox.py`, `helpers/arguments.py` | `shared/tools.py` |
| `policy/rules/`, `policy/routing.py`, `policy/facts.py` (two rule loaders) | `shared/rules.py` (one engine) |
| `policy/provider.py`, `policy/rag/`, `policy/skills.py`, `policy/citations.py` | `shared/policy/` |
| `agents/guard.py`, `agents/approver.py`, `gateway/write_gate.py`, `helpers/reply_rules.py`, `gateway/tickets.py`, `gateway/ticket_state.py`, `supervisor/escalation.py` | `shared/gates/` |
| ReAct and plan-and-execute loops in `single_agent.py`, `specialist.py`, `supervisor_agent.py` | `shared/strategies/` |
| Prompt strings in `single_agent.py`, `specialist.py`, `supervisor.py`, `approver.py`, `helpers/reply_prompt.py` | `prompts/*.md` |
| `systems/single_agent.py` | `setups/single_agentic.py` |
| `systems/supervisor_agent.py` | `setups/multi_agentic.py` |
| `systems/multi_agent.py` | `setups/multi_graph.py` |
| `agents/specialist.py`, `supervisor/supervisor.py`, `supervisor/outcome.py` | `setups/multi_common.py` |
| `gateway/route.py`, `gateway/approvals.py`, `gateway/writes.py` (second write path) | removed |
| `api.py`, `api_multi.py`, `helpers/workers.py` | `api.py` |
| `scripts/demo_app.py`, `scripts/demo_multi_agent.py` | `apps/demo.py` |
| `policy/raw_policy/pdf`, `chunk_policy`, `embed_chunk`, `skills` | `data/policy/` |
| `data/backend_seed/` | `data/db/` |
| `src/support_ops/eval/rag_evaluation.ipynb` | `notebooks/` |
| `configs/experiments/` | `configs/setups.yaml` |
| `docs/multi-agent.md`, `docs/multi-agent-flow.drawio` | `ARCHITECTURE.md`, and `docs/` per setup pair |

## Code status (2026-11-01, after the refactor on branch `refactor`)

All 19 issues listed before the refactor are resolved:

| # | Issue | Resolution |
|---|---|---|
| 1 | Two write paths | S7/S8 write through the one write gate; the route tool, approval tokens and `gateway/writes.py` are removed |
| 2 | Shared gates copied per workflow | `shared/pipeline.py::run_turn` runs guard → body → reply and reply check → ticket save for every setup |
| 3 | `plan_execute` meant three things | One react and one plan_execute loop (`shared/strategies/`) for every agent |
| 4 | Multi-agent got larger step budgets | One budget per ticket, shared by all its agents (`shared/budget.py`) |
| 5 | Only multi-agent retried | One `retries` policy, applied to every tool call and JSON reply |
| 6 | Citation check could be off for specialists | Always on, in the write gate |
| 7 | Specialists had extra tools | Every tool comes from the one registry; specialists get subsets |
| 8 | Triage in S7/S8 | Removed: the supervisor's only tool is `ask_specialist` |
| 9 | LLM cache made trials identical | Cache off by default; `run.trial` is part of the key |
| 10 | S5/S6 not built | `setups/single_graph.py` on `shared/flow.py` |
| 11 | No per-variant config | `configs/setups.yaml`; `load_config(setup="s3-c")` |
| 12 | Wrong setup labels | Code and docs use S1–S8 as in this file |
| 13 | S3/S4 subclassed S7/S8 | Four independent setup classes sharing `multi_common.py` |
| 14 | Two APIs and two demos | One `api.py` (setup per request or `run.setup`) and one `apps/demo.py` |
| 15 | Code defaults disagreed with config | Every value lives in `base.yaml`; code reads no key with a default |
| 16 | Dead or one-setup config keys | `system`, `review.trigger` and `multi_agent.*` are removed |
| 17 | Demo and experiments used different dates | `run.clock_start` everywhere; an API request can override it explicitly |
| 18 | Prompts inside code | `prompts/*.md` |
| 19 | No `.gitattributes` | `*.pdf binary` |

### Decisions made during the refactor (owners to confirm)

- **Who asks for a review in agentic setups.** The agent calls `request_review`; `backend_write` refuses an
  approver-route write until that exact write is approved. In graph setups the graph runs the review.
- **How an agentic agent ends a turn.** It calls `finish` (resolved, awaiting_customer or awaiting_human, with a
  note). An agent that stops without it is nudged once, then handed off. The S3/S4 supervisor's `reply`,
  `ask_customer` and `hand_off` became this one `finish`.
- **Ticket fields.** The ticket save in code derives the category from the turn's writes and lookups, sets
  priority "normal", and uses the first customer message as the subject. The `update_ticket` tool is gone.
- **Handoffs end the turn.** When the write gate hands a write to a person, the whole turn ends `awaiting_human`
  in every setup, so in a multi-part message later parts may not run. In the graph setups a proposal of
  `awaiting_human`, or an agent that gives up while gathering, does the same; S7/S8 otherwise end on the
  strictest outcome of their parts, or with a person if the supervisor gives up (as S3/S4 do when the supervisor
  never calls `finish`). A support request neither S7/S8 specialist covers still goes to `orders_shipping`, which
  hands it to a person, so the supervisor keeps its one tool.
- **What S7/S8's route tool did, kept for every setup.** A write may ask for a stricter route, a claimed policy
  exception gets a review, unseen returns and refunds are shown once, and code-found problems reach the reply.
  Default routes per action (exchange to the approver, account changes to a person) are not kept: those are
  unsupported writes and go to a person.
- **First return requests.** An agent may record one when the rules allow it: within the window, an allowed
  reason, and not a change of mind on final-sale goods. The agent classifies the reason from what the customer
  reported (RET-ReturnEligibility-2.3), with no approver review: no money moves on it. It is written as
  `requested`, and the store reviews the evidence and item conditions later, so those are not checked here. The refund is a separate, later step: it needs
  the return approved and inspected.
- **Replies.** Every setup's reply is written from the evidence by the `reply_writer` role; the multi-agent
  outcome summary (`tell_customer`) is removed.
- **Specialists' writes.** They are not limited per specialist; the prompts say which operations are whose.
- **Triage.** Removed as defined above (classifying the message and redirecting it). The S7/S8 supervisor still
  decides which specialist handles each part.
