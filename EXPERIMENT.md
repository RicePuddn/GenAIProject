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

| Item | Value / where |
|---|---|
| Model (all roles except the guard) | `openai/gpt-oss-120b` on Groq, temperature 0, `reasoning_effort: medium` for every role |
| Prompt attack guard | `llama-prompt-guard-2-86m`, threshold 0.5, screens every turn including history |
| Data tools | One tool registry. Multi-agent setups give each specialist a subset; they never get different implementations |
| Business rules | One rules engine, reading `src/support_ops/policy/rules/rules.json` |
| Write path | One write gate (`gateway/write_gate.py`): rule check → cited clauses must be ones retrieved in this ticket → independent approver review when required. Agentic setups reach it through the `backend_write` tool; graph setups call it as a node |
| Write review and reply check | Same approver prompts and limits (`review.max_rejections: 1`, `review.max_reply_drafts: 2`) |
| Reply | Same reply prompt and the same wording checks |
| Ticket save | Saved in code at the end of every turn, the same way in every setup |
| Budgets | 40 tool calls, 12 agent steps and US$0.05 per ticket |
| Clock | Fixed at `run.clock_start` (2026-10-01 09:00 +08:00) |

## Recommended repo structure

Organised around the setups. Everything held constant lives in `shared/`. Policy mode and strategy are config values,
so only the four combinations of agent count and control style get their own code.

```text
GenAIProject/
├── EXPERIMENT.md              study design (this file, the source of truth)
├── ARCHITECTURE.md            how the code fits together, one page
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
│   │   └── pipeline.py        run_turn(): guard → setup body → reply check → save ticket
│   ├── prompts/               one .md per role: agent, supervisor, specialists, approver, reply writer
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
│   └── setups/                behaviour tests: message in → expected outcome and writes, for every setup
└── results/                   run outputs (gitignored except summaries)
```

Rules for the structure:

- A setup file imports only from `shared/`, `prompts/` and `multi_common.py`, never from another setup file.
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
| `docs/multi-agent.md`, `docs/multi-agent-flow.drawio` | `ARCHITECTURE.md` |

## Current code status (2026-11-01)

### A. Skews the comparison

These make setups differ in ways other than the factor being tested.

1. **Two write paths.** The fixed multi-agent workflow writes through `route` → approval tokens (`gateway/approvals.py`,
   `gateway/writes.py`). Every other setup writes through `gateway/write_gate.py`. Move S7/S8 onto the write gate.
2. **The shared gates are copied rather than shared.** Guard, reply check and ticket save are re-implemented in each
   workflow. Wrap every setup as `guard → setup body → reply check → save ticket`, so setups implement only the body.
3. **`plan_execute` means three different things.**
   - Single agent: plans every call including writes, then gets up to 3 completion checks and 2 replans (`systems/single_agent.py`).
   - Specialists: plan only reads and policy calls, then write one proposal (`agents/specialist.py`).
   - Supervisor: plans a list of tasks, one per specialist.

   Implement `react` and `plan_execute` once and use the same loop for every agent.
4. **Multi-agent setups get far larger step budgets.** The single agent has 12 agent steps per ticket. Each specialist
   run gets its own 12 (`agents/specialist.py:65`), the agentic supervisor another 12 (`max_supervisor_steps`) and the fixed
   workflow 4 delegations (`max_orchestration_steps`). Only the 40-tool-call and US$0.05 caps are shared, and the cost cap
   cuts off multi-agent setups sooner because they make more LLM calls. Budgets should be per ticket and equal.
5. **Only the multi-agent setups retry.** Specialists re-run a failed tool once and re-ask after malformed JSON
   (`multi_agent.retries`). The single agent does not retry failed tools. Use one retry policy for every setup.
6. **The citation check can be switched off for specialists only** (`multi_agent.check_citations`). The single agent
   always checks. Make it fixed and on for every setup.
7. **Specialists get tools the single agent does not have** (for example the extra returns lookup in `agents/toolbox.py`).
   Every tool should come from the one registry, with specialists getting subsets.
8. **Triage still exists in the fixed multi-agent workflow** (the supervisor classifies and can redirect). Remove it for S7/S8.
9. **The LLM response cache makes repeated trials identical.** The cache key is the exact request (`helpers/llm.py:102`),
   with temperature 0 and a fixed seed. With the cache on (the default, `llm.cache_dir: results/cache`), repeat trials replay
   the first one, so every setup looks perfectly consistent (H2), and cached calls take no time and cost nothing (H1).
   Turn the cache off for experiment runs, or add the trial number to the key.

### B. Missing

10. **S5/S6 are not built.**
11. **No per-variant config files.** Only `multi_agent_rag.yaml` and `multi_agent_skills.yaml` exist. Define the 24
    variants, ideally as one matrix (`configs/setups.yaml`) that a setup ID such as `s3-c` resolves from, not 24 hand-written files.

### C. Confusing code and config

12. **The setup labels in the code are wrong.** `systems/multi_agent.py` and `docs/multi-agent.md` call the fixed
    workflow "S3/S4". In this plan it is S7/S8. `systems/supervisor_agent.py` is the agentic S3/S4.
13. **The agentic multi-agent class inherits from the fixed one.** `SupervisorAgentWorkflow` (S3/S4) subclasses
    `MultiAgentWorkflow` (S7/S8) and replaces its graph. Separate them, sharing code through the common pieces instead.
14. **Two APIs and two demo apps** (`api.py` on port 8000, `api_multi.py` on 8001) with different request schemas.
    Merge them into one, with the setup and variant chosen by config.
15. **Code defaults disagree with the config.** The single agent falls back to `policy.mode: skills` while `base.yaml` says
    `rag`. `api_multi.py` falls back to `design: workflow` while the configs say `supervisor`. A missing key silently
    gives a different setup. Defaults should live only in `base.yaml`, and a missing key should be an error.
16. **Config keys that do nothing or apply to one setup only.** `system: multi_agent` is read by no code.
    `review.trigger` is read only by the agentic multi-agent workflow. Remove them or make them apply everywhere.
17. **The demo and the experiments use different dates.** Both APIs use today's date, while programmatic runs use
    `run.clock_start`. Return-window answers in the demo differ from experiment runs, and seeded orders drift out of
    their 30-day windows as real time passes. Use `run.clock_start` everywhere, with an explicit override for the demo.
18. **Prompts are long strings inside the code.** Move each role's prompt to its own file so it can be read and compared across setups.

### D. Environment and repo

19. **No `.gitattributes`.** On Windows, `core.autocrlf` rewrites the policy PDFs and breaks the rules fingerprint
    check. Add `*.pdf binary`.