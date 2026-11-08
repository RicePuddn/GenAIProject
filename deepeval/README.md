# DeepEval evaluation workflow

Collect customer-support outputs first, then score the saved traces with DeepEval. Collection calls the application and changes the evaluation database. Scoring reads saved outputs and uses paid OpenRouter judge calls without calling the support agent or its database.

Run the commands below from the repository root. Examples use Bash and `.venv/bin/python`; on Windows, use `.venv/Scripts/python.exe` or run the Bash wrapper in an appropriate shell.

## Case files

| File                                                          | Purpose                                                                                                   |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `deepeval/cases_full.csv`                                     | All 13 original evaluation questions, expected answers/statuses, tool expectations, and multi-turn inputs |
| `results/deepeval/robustness_evaluation/cases_paraphrase.csv` | Paraphrased versions of the 13 cases                                                                      |
| `results/deepeval/robustness_evaluation/cases_messy.csv`      | Messy-input versions of the 13 cases                                                                      |
| `results/deepeval/case_categories.csv`                        | Case descriptions and category labels for analysis                                                        |

The 13 questions cover these categories:

| Case    | Category               | Focus                                |
| ------- | ---------------------- | ------------------------------------ |
| EVAL-01 | Simple lookup          | Order contents                       |
| EVAL-02 | Simple lookup          | Order status                         |
| EVAL-03 | Simple lookup          | Payment status                       |
| EVAL-04 | Simple lookup          | Return status                        |
| EVAL-05 | Simple lookup          | Latest tracking status               |
| EVAL-06 | Simple lookup          | Return window policy                 |
| EVAL-07 | Simple lookup          | Identify a processing order          |
| EVAL-08 | Simple lookup          | Shipment carrier and tracking number |
| EVAL-09 | Multi-step             | Refund processing                    |
| EVAL-10 | Multi-step             | Leaking-battery safety escalation    |
| EVAL-11 | Safety and guardrail   | Malicious instruction injection      |
| EVAL-12 | Multi-step, multi-turn | Clarify the order before tracking    |
| EVAL-13 | Multi-step, multi-turn | Prevent duplicate cancellation       |

`case_id` and `ticket_id` are separate identifiers; do not assume their numeric suffixes match. For multi-turn cases, the collector sends the messages in `Turns` and carries the returned conversation history into the next turn. Editing only `Query` does not change a case whose `Turns` field is populated.

## Prerequisites

Use the project's existing virtual environment with the application dependencies, `requests`, `psycopg`, and DeepEval installed. Configure `.env` with:

- `SUPPORT_OPS_DATABASE_URL`: a dedicated evaluation database.
- `OPENROUTER_API_KEY`: required for the current OpenRouter application profile and scoring judge.
- `PINECONE_API_KEY` and the configured retrieval resources for RAG variants.

The API and collector must use the same database. Start the database and API as described in the root README. For the API:

```bash
.venv/bin/uvicorn SupportOpsBot.chatRoute:app --host 127.0.0.1 --port 8000
```

The current application profile uses `openai/gpt-oss-120b` through OpenRouter. Restart the API after changing its provider configuration. The collector's `--judge-model` flag only records a label in the collected output; it does not select the application's model. The scorer's `--judge-model` selects the evaluation judge.

### Database reset behavior

Before every **case × setup** combination, collection:

1. Removes refunds and returns identified by that ticket's recorded writes.
2. Deletes that ticket's write records and support-ticket record.
3. Upserts the shared canonical seed fixtures, including orders, items, and shipments.

This restores seeded records and performs ticket-specific cleanup; it does not truncate the entire database. Seed upserts also update existing fixture rows, so use a dedicated evaluation database. Do not run collection batches concurrently against the same database: their resets and writes can interfere.

A multi-turn case is reset once before its first turn. State is preserved between the turns in that case. Preflight checks customer IDs against the API's customer list before collection begins. A dry run stops before API calls or database changes.

## Collect all 13 cases across 24 variants

Preview the selection:

```bash
bash deepeval/collect_full.sh --dry-run
```

Collect fresh outputs:

```bash
caffeinate -i bash deepeval/collect_full.sh
```

The wrapper selects all 24 variants, giving **312 case/setup records**. Multi-turn cases make more than one request, so this is not a count of individual API calls. Collection is sequential. The wrapper creates a fresh directory such as:

```text
results/deepeval/rerun_YYYYMMDD_HHMMSS_PID/results.jsonl
```

It prints the exact output path and the corresponding scoring command. `caffeinate -i` is optional on macOS; it prevents idle sleep while the command runs. Locking the screen is fine. Keep the lid open and the network connected.

The JSONL contains final replies, actual and expected statuses, reference answers, tool calls and arguments, writes, and ordered execution events. Each completed case/setup record is flushed to disk. Collection appends to its output file; use a fresh output path for a new round. Reusing a partially filled output path does not skip completed records and can introduce duplicates.

### Select a case or API endpoint

The wrapper forwards extra arguments to the collector:

```bash
bash deepeval/collect_full.sh --case EVAL-13
bash deepeval/collect_full.sh --api http://127.0.0.1:8000
```

To collect a single case and variant, call the Python runner directly:

```bash
.venv/bin/python deepeval/run.py \
  --cases deepeval/cases_full.csv \
  --case EVAL-13 \
  --setup s1-a \
  --judge-model openai/gpt-oss-120b \
  --output results/deepeval/single_case/results.jsonl
```

When invoking `run.py` directly, always specify `--cases`. Its legacy default points to `deepeval/cases.csv`, which is not the full case source and may be absent. Without explicit `--setup` arguments, the Python runner selects only `s1-a`. The Bash wrapper supplies all 24 variants.

## Collect robustness variants

```bash
bash deepeval/collect_full.sh \
  --cases results/deepeval/robustness_evaluation/cases_paraphrase.csv \
  --dry-run

caffeinate -i bash deepeval/collect_full.sh \
  --cases results/deepeval/robustness_evaluation/cases_paraphrase.csv
```

This uses the same collection and reset process with a fresh output directory. Substitute `cases_messy.csv` to collect the messy-input variants.

For a one-word robustness test, create a separate copy of `cases_full.csv`, such as `deepeval/cases_robustness_one_word.csv`, and make one meaning-preserving word edit per initial question. Pass that file with `--cases` after creating it; it is not present in the current checkout. For multi-turn cases, also change the initial message in `Turns`, preserving the follow-up messages. Keep the baseline CSV unchanged.

Before collection, check that the baseline and robustness files use the same current reference answers, expected statuses, and tool expectations. Keep those aligned when correcting a case's ground truth. Compare matching case/setup records under the original and modified wording. Robustness concerns maintaining a correct outcome and appropriate tool use; identical response wording or an identical trajectory is not required.

## Score saved outputs

Set `run_dir` to the directory printed by collection, or to an existing round such as `results/deepeval/rerun2`:

```bash
run_dir="results/deepeval/rerun2"

caffeinate -i .venv/bin/python deepeval/evaluate.py \
  --input "$run_dir/results.jsonl" \
  --output "$run_dir/scored.csv" \
  --judge-model openai/gpt-oss-120b
```

Paths must match the actual directory: `results/deepeval/rerun2/results.jsonl` differs from `results/rerun2/results.jsonl`. The judge reads `OPENROUTER_API_KEY` from the environment or `.env`. Scoring writes one CSV row per selected saved record, including metric scores, automatic success flags, and judge explanations. It **overwrites** the specified output CSV; choose a separate filename for retries or revised references.

Available scoring filters include repeatable `--case` and `--setup` options, plus `--limit`. For example:

```bash
.venv/bin/python deepeval/evaluate.py \
  --input "$run_dir/results.jsonl" \
  --output "$run_dir/scored_subset.csv" \
  --case EVAL-13 --setup s1-a \
  --judge-model openai/gpt-oss-120b
```

### Metrics

The scorer uses:

- **Answer correctness:** G-Eval against the saved reference answer.
- **Tool correctness:** DeepEval's tool-correctness measure with expected calls and ordering.
- **Argument correctness:** DeepEval's argument-correctness measure.
- **Task completion, step efficiency, and trajectory quality:** G-Eval rubrics over recorded workflow events.
- **Plan adherence and plan quality:** trace-based G-Eval when the saved trace identifies a plan-and-execute strategy.

The trajectory criteria are custom G-Eval adaptations over application traces, rather than direct use of DeepEval-native trajectory metric classes. Graph writes appear as `backend_write` in the trace and corresponding tool expectations.

The analysis notebooks separately derive task pass from reviewed answer correctness plus exact ticket-status agreement, and tool-call pass from reviewed tool and argument correctness. Manual pass/fail review is separate from the scorer's numeric scores and automatic success flags.

See [DeepEval's AI Agent evaluation metrics](https://deepeval.com/guides/guides-ai-agent-evaluation-metrics) and [Tool Correctness](https://deepeval.com/docs/metrics-tool-correctness).

## Re-score a case after correcting its ground truth

The scorer uses the `ground_truth` stored in its **input JSONL**. Updating the case CSV alone does not update existing collected records. For an answer-reference correction, preserve the recorded responses and traces and update the corresponding saved reference in a separate JSONL copy. Recollect only if you need fresh agent behavior, such as after changing the question, fixture state, or application.

If the saved JSONL already contains the corrected reference, re-score the selected case directly. For EVAL-13 across rerun2 and rerun3:

```bash
for round in rerun2 rerun3; do
  caffeinate -i .venv/bin/python deepeval/evaluate.py \
    --input "results/deepeval/$round/results.jsonl" \
    --output "results/deepeval/$round/scored_eval13_retry.csv" \
    --case EVAL-13 \
    --judge-model openai/gpt-oss-120b
done
```

For EVAL-11, change the filter to `--case EVAL-11` and the output filename to `scored_eval11_retry.csv`. Each case selects 24 saved variants per complete round. Both cases can be selected together by supplying `--case EVAL-11 --case EVAL-13` and choosing a distinct output filename.

After scoring, merge replacements into a **new** combined score file using the exact key `(case_id, setup, ticket_id)`. Replace only the matching records, verify that each replacement key is unique, and preserve all other rows. Review any manual judgments affected by the reference change. Keep the original scores for comparison.

## Retry judge errors

An `ERROR:` in a metric's reason column indicates a scoring failure, such as an invalid JSON response from the judge. It is different from a valid low score or failed task. The scorer attempts the remaining metrics and records errors in the CSV; its final exit status is nonzero if any metric raised an error.

Match affected CSV records to the source JSONL using `(case_id, setup, ticket_id)` and save only those traces in a separate retry JSONL. If rerun3's previously extracted retry file exists, score it with:

```bash
caffeinate -i .venv/bin/python deepeval/evaluate.py \
  --input results/deepeval/rerun3/results_errors_retry.jsonl \
  --output results/deepeval/rerun3/scored_retry.csv \
  --judge-model openai/gpt-oss-120b
```

That file is a prepared subset, not an automatically generated scorer output. Rebuild it from the relevant score file when the set of errors changes. Retrying a saved record evaluates all its applicable metrics, not just the failed metric. Merge successful replacements by the same three-part key, retaining the original score file.

### Resume interrupted scoring

`--start-at CASE,SETUP,TICKET` starts at the matching saved record, including that record, and continues through the remaining selected input. Use a new output filename because the scorer does not append:

```bash
.venv/bin/python deepeval/evaluate.py \
  --input "$run_dir/results.jsonl" \
  --output "$run_dir/scored_resume.csv" \
  --start-at EVAL-02,s3-b,TKT-EVAL-03 \
  --judge-model openai/gpt-oss-120b
```

Use the exact key of the first record to re-score. Merge the completed prefix and resumed records afterward, checking keys for duplicates.
