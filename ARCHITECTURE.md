# Architecture

How the code fits together. The study design (setups, variants, what is held constant) is in
[EXPERIMENT.md](EXPERIMENT.md); install and run steps are in [README.md](README.md).
Each setup's walkthrough and draw.io diagram is in [docs/](docs/README.md).

## One turn

Every setup runs the same pipeline, `shared/pipeline.py::run_turn`:

```text
guard ─► setup body ─► reply writer ─► wording checks ─► reply check ─► ticket save
  │          │                                               │
  │          └─ budget hit ──────────────────────┐           └─ no draft passes ─┐
  └─ flagged ─────────────────────────────────── hold reply + handoff ◄─────────┘
```

A setup implements only its body (`setups/*.py`) and returns a `TurnResult`: a status
(`resolved`, `awaiting_customer` or `awaiting_human`), the agent's note, and, for multi-agent
setups, each specialist's report. Everything the agents did is in the turn's `TurnContext`
(`shared/context.py`): the evidence (every data tool call, in order), the writes, the trace and the
budget. The guard, the reply and the ticket save never depend on which setup ran.

If the LLM provider fails at any step, `run_turn` ends the turn `in_progress`: it saves the ticket and sends a
fixed reply asking the customer to try again. A tool call that fails for good and never works later does the
same to a turn that would otherwise end `resolved`, whatever the agent concluded. Sending the message again
on the same ticket is safe, because each write replays by its idempotency key.

## The four code paths

| File | Setups | Body |
|---|---|---|
| `setups/single_agentic.py` | S1, S2 | One agent with every tool; it finishes by calling `finish`. |
| `setups/single_graph.py` | S5, S6 | `shared/flow.py` with the agent in the gather stage. |
| `setups/multi_agentic.py` | S3, S4 | The supervisor calls `ask_specialist`; each specialist acts like S1's agent on its part and finishes with `finish`. |
| `setups/multi_graph.py` | S7, S8 | The supervisor calls `ask_specialist`; each part runs `shared/flow.py` with the specialist in the gather stage. No triage. |

`setups/multi_common.py` holds what the multi-agent setups share: the specialists' tool subsets,
policy scopes and prompts, and the `ask_specialist` tool. A specialist's `policy` tool is the same provider
as the single agent's, seen through `scoped(...)`: by default it keeps to the specialist's own sections
(RAG) or skill folders (skills), and a call with `scope: all` reaches the rest. A setup file imports only from `shared/`, `prompts/` and
`multi_common.py`. `setups/__init__.py::build_setup("s3-c")` picks the class from the setup's agents
and control style; the policy mode and strategies are config.

## Control styles

**Agentic.** The agent holds `backend_write` and `request_review` and decides when to use them.
A write on the approver route is refused (`review_required`) until `request_review` has approved that
exact write.

**Graph.** `shared/flow.py` is a LangGraph:

```text
gather ─► propose ─► check ─► review (approver route) ─► write ─► next write … ─► done
  ▲                    │         │                         │
  └────────────────────┴─ refused, with the reason ────────┘
```

The agent only gathers (read and policy tools), then returns a `Proposal` (structured output:
outcome, note, writes, review_reason). Code runs the write gate on each write. A block, a fixable
review rejection or a refused citation sends the case back to gather; a handoff ends the turn.

## Strategies

`shared/strategies/` has one loop per strategy, used by every agent: the single agent, the supervisor
and the specialists. An agent is a prompt plus a `Toolset`.

- `react`: one tool call per turn. With a terminal tool (`finish`) the agent is nudged once if it
  stops without it; without one, it ends when it stops calling tools.
- `plan_execute`: plan every call, run the plan, and stop at the first error, `blocked` or
  `review_required` to replan (`limits.max_replans`). After a plan runs in full, a completion check
  asks for any missing step (`limits.max_plan_checks`).

## Held constant, and where

| What | Where |
|---|---|
| Tools | `shared/tools.py`: one `ToolRegistry`; `Toolset.call` applies the budget, tracing and evidence for every tool call; `run_with_retries` is the one retry policy, also used by the graph's write stages. |
| Rules | `shared/rules.py`: `Rules` (from `data/policy/rules.json`) and `BackendFacts` (recorded facts, never the model's). |
| Writes | `shared/gates/write_gate.py`: `WriteGate` decides (rules, then citations, then review), `WriteExecutor` writes once per idempotency key. |
| Review and reply check | `shared/gates/approver.py`, one prompt each in `prompts/approver.md`. |
| Reply | `shared/gates/reply.py`: the reply writer (`prompts/reply_writer.md`), code wording checks, then the reply check. |
| Guard | `shared/gates/guard.py`: screens the new customer message once per turn. |
| Ticket save and handoff | `shared/gates/ticket.py`, `shared/gates/handoff.py`, in code at the end of every turn. |
| Budgets and retries | `shared/budget.py`: one budget per ticket, shared by every agent in it. |
| Policy | `shared/policy/`: the `policy` tool's RAG and skills providers and the citation check. |
| Database | `shared/db/`: read-only lookups, fixed write transactions, the signed-in account's view. |

## Prompts

`prompts/*.md`, loaded with `prompts.load(name, section, **values)`: `agent.md`, `specialists.md`,
`supervisor.md`, `approver.md`, `reply_writer.md`, plus what several roles share: `facts.md` (how to
check facts and policy), `control.md` (agentic or graph), `strategies.md` and `propose.md`. A
placeholder left unfilled, or a value with no placeholder, is an error.

## Config

`configs/base.yaml` holds every value held constant; code reads no key with a default.
`configs/setups.yaml` is the matrix: `load_config(setup="s3-c")` adds `setup` (name, agents, control,
variant), `policy.mode` and `strategy` (each role's react or plan_execute). `configs/rag/` holds the
retrieval profiles the RAG notebook compares.

## Entry points

- `api.py`: `/chat` and `/chat/stream` for any setup (`setup` in the request, else `run.setup`).
- `apps/demo.py`: a Streamlit page over the API.
- `scripts/`: `seed_db.py`, `build_rag_index.py`, `sync_skills.py`.
- `notebooks/rag_evaluation.ipynb`: retrieval evaluation; results in `results/rag/`.

## Tests

`tests/fakes.py` has a scripted LLM, an in-memory database and a fake retriever, so no test needs an
API key, PostgreSQL or Pinecone. `tests/setups/` runs each behaviour scenario in all 24 variants from
one description (`tests/setups/scripts.py`). `tests/shared/` unit-tests the gates, rules, tools,
strategies and policy. Tests against a real database run when `SUPPORT_OPS_TEST_DATABASE_URL` is set.
