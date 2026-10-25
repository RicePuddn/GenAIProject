# Skills-style policy (skills arms: S2, S4)

The same policy content as `policy/corpus/`, organised for progressive disclosure:

```
<size>/                       # 5p/, 20p/, 50p/
  <skill-name>/               # e.g. returns-and-refunds/
    SKILL.md                  # frontmatter `name` + one-line `description`, then an overview
                              # pointing to the clause files below
    <clause files>.md         # same clause ids as the corpus
```

The agent first sees only each skill's description (`list_skills`), opens one `SKILL.md`
(`open_skill`), then reads specific clause files (`read_policy_file`).
Record the time spent organising each size by hand: the report compares authoring effort.
