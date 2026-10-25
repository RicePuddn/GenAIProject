# Policy-Bound Multi-Agent Customer Support Operations

IS469 Generative and Agentic AI group project. We compare single-agent and multi-agent
support systems, each with RAG or skills-style policy retrieval. In every system, LLMs propose refunds,
returns and order changes, and deterministic code (rule checks, gateway, approval tokens) decides whether they happen.

> Status: project setup only (layout, config, shared helpers). Components are in progress.

## 1. Environment setup

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env   # then add GROQ_API_KEY
```

## 2. Tests

```bash
.venv/bin/python -m pytest -q
```

## Layout

```
configs/            base.yaml (models, seeds, limits) and experiments/*.yaml
policy/             corpus/ (RAG arms), skills/ (skills arms)
data/
  cases/            labelled cases
  backend_seed/     seeded background data for the mock backend
src/support_ops/
  helpers/          clock, config, tracing, LLM client (shared plumbing)
  policy/           hard-rule checks, RAG retriever, skills navigator
  backend/          mock database, lookups, write endpoints
  gateway/          route check, approval tokens, ticket state, audit log
  agents/           specialists and the approver / QA agent
  supervisor/       ReAct and plan-and-execute supervisors
  systems/          one module per arm (S1-S4)
  eval/             harness, metrics, results tables
scripts/            command-line entry points
tests/              automated tests
results/            raw traces and final tables (raw/ and cache/ are git-ignored)
```

Each folder's `__init__.py` or `README.md` says what goes in it and who owns it.

Sections to add as components land: obtaining data and models, building indexes, running
experiments, regenerating results tables, expected cost and runtime.
