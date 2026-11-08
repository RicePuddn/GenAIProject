# DeepEval results

Evaluation results for the 24 agent setups on the 13 test cases.
The test cases themselves live in [`deepeval/cases/`](../../deepeval/cases/).

## Folder layout

**`data/` → `analysis/` → `output/`.** Everything in `data/` is *input* to the notebooks; results live only in `output/`.

| Folder | What's in it | Edit by hand? |
|---|---|---|
| `data/raw/` | Raw output of each evaluation run: `results.jsonl` (agent logs) and `scored.csv` (DeepEval judge scores) | Never |
| `data/prepared/` | Analysis input, not results: each raw run standardised to one row per case × setup, with the same columns for every run | No, regenerate instead |
| `analysis/` | Notebooks numbered in run order, plus `helpers.py` (shared loading, tests and chart style) | — |
| `output/tables/` | Result tables used in the report | No, regenerate instead |
| `output/figures/` | Charts used in the report | No, regenerate instead |
| `archive/` | Files kept for reference that nothing reads | — |

### Runs

| Run | Questions | Raw output | Prepared file |
|---|---|---|---|
| Baseline run 1 | `cases/baseline.csv` | `data/raw/baseline/run1/` | `data/prepared/baseline_run1.csv` |
| Baseline run 2 | `cases/baseline.csv` | `data/raw/baseline/run2/` | `data/prepared/baseline_run2.csv` |
| Baseline run 3 | `cases/baseline.csv` | `data/raw/baseline/run3/` | `data/prepared/baseline_run3.csv` |
| Paraphrased run 1 | `cases/paraphrased.csv` | `data/raw/paraphrased/` | `data/prepared/paraphrased_run1.csv` |

Prepared files are generated from the raw runs with `analysis/prepare_run.py` (same columns and rules as the baseline: `*_success_manual` = score ≥ 0.5; tokens summed over `llm_call` events; `elapsed_s` summed over `workflow_end` events). Running it on `data/raw/baseline/run2` reproduces `baseline_run2.csv` exactly.

- **Baseline** is the original wording of the 13 questions.
- **Paraphrased** has the same questions, reworded, with the same expected answers, statuses and tool calls.

New runs are collected with `bash deepeval/collect_full.sh --variant <baseline|paraphrased>`. The script writes to the next free `data/raw/<variant>/runN/` folder. See [`deepeval/README.md`](../../deepeval/README.md).

`data/raw/baseline/run1/scored_dedup.csv` exists only for run 1. Run 1 was scored in batches and some cases were retried, so `scored.csv` holds 337 rows. The dedup file keeps one row per case and setup (312 rows) and adds tokens, latency and ticket status from the logs. Runs 2 and 3 had no retries, so they don't need this file.

## Three ways the setups are measured

| Analysis | Question it answers | Data used |
|---|---|---|
| **Scorecard** | How good is each setup? | Baseline run 1, plus a 3-run average |
| **Consistency** | Run the same setup 3 times: do you get the same pass/fail each time? | Baseline runs 1, 2 and 3 |
| **Robustness** | Does the setup still pass when the question is reworded? | Baseline run 1 and paraphrased run 1 |

Consistency and robustness are both about reliability, but they vary different things. Consistency keeps the question fixed and repeats the run. Robustness keeps the run fixed and changes the question.

The three quality metrics, used throughout:

- **Task success**: the answer was judged correct *and* the ticket ended in the expected status.
- **Tool-call accuracy**: the right tools were called *and* with the right arguments.
- **Step efficiency**: the agent got there without unnecessary steps.

## Notebooks (`analysis/`)

Run them in order: `3_scorecard` reads the outputs of the first two, and `5_findings` reads all of them.

| Notebook | Reads | Writes |
|---|---|---|
| `1_consistency.ipynb` | `data/prepared/baseline_run{1,2,3}.csv` | `output/tables/consistency_*.csv` |
| `2_robustness.ipynb` | `data/prepared/baseline_run1.csv`, `data/prepared/paraphrased_run1.csv` | `output/tables/robustness_by_setup.csv`, `robustness_by_setup_and_case.csv` |
| `3_scorecard.ipynb` | `data/prepared/`, `output/tables/consistency_*`, `output/tables/robustness_*` | `output/tables/scorecard_*.csv` |
| `4_hypothesis_tests.ipynb` | `data/raw/baseline/run1/scored_dedup.csv` | `output/figures/` (matched-pair significance tests for H1–H4; statistical backup) |
| **`5_findings.ipynb`** | everything above | `output/figures/findings_*.png`. **Results and Discussion: the presentation notebook** |

`5_findings_script.md` holds the talking points for presenting `5_findings`, with headings that match its sections.
`helpers.py` holds the set-up code shared by notebooks 4 and 5, so the notebooks show only analysis.

## Tables (`output/tables/`)

File names follow the pattern **`<analysis>_by_<one row per>`**.

### Scorecard: the headline results

| File | One row per | Contents |
|---|---|---|
| **`scorecard_by_setup.csv`** | setup (24) | **Main results table.** All quality metrics, median cost and latency, plus the consistency % and robustness % from the tables below. |
| `scorecard_by_category.csv` | architecture group × question category | The same metrics split by question type (simple lookup / multi-step / safety). Shows which kinds of question are hard for which architectures. |
| `scorecard_columns.csv` | column of the scorecard | Glossary: what each scorecard column means and which file it comes from |

### Consistency

| File | One row per | Contents |
|---|---|---|
| `consistency_by_setup.csv` | setup (24) | % of the 13 cases with the same pass/fail in all 3 runs, and % with cost stable within 10% and latency within 20% |
| `consistency_by_setup_and_case.csv` | setup × case (312) | Drill-down: did this exact case give the same result in every run? |
| `consistency_pass_rates_by_run.csv` | run (3), plus mean and range | Overall pass rates of run 1 vs 2 vs 3, showing how much the totals move between runs |
| `consistency_changes_by_case.csv` | case (13) | How many of the 24 setups changed outcome on this case. Shows which questions are unstable. |
| `consistency_by_architecture.csv` | architecture group | Consistency % for each design choice (single vs multi-agent, graph vs agentic, RAG vs skills, ReAct vs plan-execute) and each question category |

### Robustness

| File | One row per | Contents |
|---|---|---|
| `robustness_by_setup.csv` | setup (24) | Pass rates on the baseline and paraphrased questions, plus the **robust** rate and **retention** (see below) |
| `robustness_by_setup_and_case.csv` | setup × case (312) | Drill-down: did this exact case pass on both wordings? |
| `robustness_question_variants.csv` | case (13) | The baseline and paraphrased wording side by side, for showing in the report what was changed. Static: no notebook writes it, so update it by hand if the case files change. |

- **Robust rate**: % of the 13 cases that passed on *both* wordings.
- **Retention**: of the cases a setup passed on the baseline, the % that still passed when reworded. This separates how sensitive a setup is to wording from how good it is overall. For example, a setup that passes 11/13 on the baseline and keeps 10 of those has a robust rate of 76.9% and retention of 90.9%.

## Renamed files

| Old name | New name |
|---|---|
| `final_evaluation/final_evaluation_summary.csv` | `output/tables/scorecard_by_setup.csv` |
| `final_evaluation/final_evaluation_by_category.csv` | `output/tables/scorecard_by_category.csv` |
| `final_evaluation/final_evaluation_columns.csv` | `output/tables/scorecard_columns.csv` |
| `consistency_evaluation/final_consistency_summary.csv` | `output/tables/consistency_by_setup.csv` |
| `consistency_evaluation/case_consistency_per_case.csv` | `output/tables/consistency_by_setup_and_case.csv` |
| `consistency_evaluation/consistency_by_run.csv` | `output/tables/consistency_pass_rates_by_run.csv` |
| `consistency_evaluation/consistency_by_case.csv` | `output/tables/consistency_changes_by_case.csv` |
| `consistency_evaluation/consistency_by_group.csv` | `output/tables/consistency_by_architecture.csv` |
| `robustness_evaluation/final_robustness_summary.csv` | `output/tables/robustness_by_setup.csv` |
| `robustness_evaluation/case_robustness_per_case.csv` | `output/tables/robustness_by_setup_and_case.csv` |
| `robustness_evaluation/robustness_cases_overview.csv` | `output/tables/robustness_question_variants.csv` |
| `analysis/architecture_analysis.ipynb` | `analysis/4_hypothesis_tests.ipynb` |
| `results-and-discussions.ipynb` | `analysis/5_findings.ipynb` |
| `clean_scored_merged_dedup.csv` | `data/prepared/baseline_run1.csv` (original Excel kept in `archive/`) |
| `consistency_evaluation/run{2,3}_clean_scored_merged_dedup_.csv` | `data/prepared/baseline_run{2,3}.csv` |
| `robustness_evaluation/paraphrase_clean_scored_merged_dedup_.csv` | `data/prepared/paraphrased_run1.csv` |
| `scored_merged_dedup.csv` | `data/raw/baseline/run1/scored_dedup.csv` |
| `run1/`, `rerun2/`, `rerun3/` | `data/raw/baseline/run{1,2,3}/` |
| `case_categories.csv` | `deepeval/cases/categories.csv` |
| `figures/` | `output/figures/` |
| `runs/`, `clean/` | `data/raw/`, `data/prepared/` |
