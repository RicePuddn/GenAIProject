# Skills-style policy retrieval

The skills setups (S2, S4, S6, S8) use these fictional policy skills:

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

The `check` action runs the delivery rules for an order's parcels, from
[`rules.json`](../rules.json), with the recorded shipment facts. Writes need no check: `backend_write`
applies the rules itself.

Configure `policy.skills_dir` to select another skill set. Relative paths resolve
from the repository root. For future RAG/skills comparisons, keep the same policy
clauses in the corpus and skill variants, and record policy authoring effort.
