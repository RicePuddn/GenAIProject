# Skills-style policy retrieval

The single-agent flow currently uses these fictional policy skills:

```text
orders-and-shipping/
  SKILL.md
  tracking.md
  changes.md
returns-and-refunds/
  SKILL.md
  eligibility.md
  refunds.md
```

The `policy` tool supports progressive disclosure:

1. `{"action":"list"}` returns skill names, descriptions and available filenames.
2. `{"action":"open","skill":"returns-and-refunds"}` reads its `SKILL.md` overview.
3. `{"action":"read","skill":"returns-and-refunds","file":"eligibility.md"}` reads the clause.

The tool also accepts a skill directory as `file` when reading an entrypoint.
`policy.md`, `skill.md` and `index.md` are aliases for `SKILL.md` within a selected
skill. Unknown clause filenames return an error listing available files so the
agent can correct its call. Paths must stay inside the selected skill directory.

The `check` action uses the same deterministic rules as `ticket_state`, defined in
[`rules.json`](../../src/support_ops/policy/rules/rules.json). Supply recorded facts
with the operation and amount; missing facts or mandatory-review cases need a human.

Configure `policy.skills_dir` to select another skill set. Relative paths resolve
from the repository root. For future RAG/skills comparisons, keep the same policy
clauses in the corpus and skill variants, and record policy authoring effort.
