# Policy corpus (RAG arms: S1, S3)

The policy documents the RAG retriever searches. Planned versions at three sizes: 5, 20 and 50 pages,
one folder each (`5p/`, `20p/`, `50p/`), selected by `policy.corpus_size` in `configs/base.yaml`.

Owner: Peiying.

`mini/` is a stand-in until those exist: the same clause text as `policy/skills/`, with clause ids,
so S1 and S2 read identical policy content.

## Format the loader expects

Each folder holds `.md` or `.pdf` files. With `policy.rag.chunking: clause`, every clause is one chunk
and starts with a line holding its id and title; in markdown that line is a heading:

```markdown
---
section: Returns and refunds          # optional; otherwise the first "# " heading or the file name
---
# Returns and refunds

## RET-1 Return window
Customers may request a return within 30 days of delivery...

## RET-1-EX-1 Final-sale items
Exception to: RET-1
Marked final-sale items are not returnable.
```

- Clause ids are upper-case letters and numbers joined by hyphens, ending in a number
  (`RET-1`, `RET-1-EX-1`, `SHP-12`), and must be unique across the folder.
- `Exception to: <id>[, <id>]` marks an exception. When a search returns a clause, the retriever
  also returns its exceptions.
- In a PDF, keep each clause id at the start of its own line so it survives text extraction.

With `policy.rag.chunking: window`, files are split into overlapping word windows instead
(Lab B style), with no clause ids or exception links.
