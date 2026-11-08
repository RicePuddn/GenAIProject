"""Standardise one raw evaluation run into data/prepared/, with the same columns as the baseline files.

Usage (from results/deepeval/):
    python analysis/prepare_run.py data/raw/paraphrased  data/prepared/paraphrased_run1.csv --variant paraphrase
    python analysis/prepare_run.py data/raw/baseline/run2 data/prepared/baseline_run2.csv   --run-id run_2

Input:  <raw_dir>/scored.csv (DeepEval judge scores) and <raw_dir>/results.jsonl (agent logs).
Output: one row per case x setup (312), columns and row order identical to data/prepared/baseline_run1.csv.

Derived columns, using the same rules as the baseline files:
  *_success_manual  score >= 0.5 (DeepEval's default threshold), replacing the judge's unreliable *_success flag
  category          copied from baseline_run1.csv by case_id
  input_tokens      sum of input tokens over every llm_call event in the log
  output_tokens     sum of output tokens over every llm_call event
  elapsed_s         sum of elapsed_s over every workflow_end event (multi-turn cases have one per turn)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

DEEPEVAL = Path(__file__).resolve().parents[1]
BASELINE = DEEPEVAL / "data/prepared/baseline_run1.csv"
KEY = ["case_id", "setup"]
METRICS = ["answer_correctness", "tool_correctness", "argument_correctness", "step_efficiency_trace"]
THRESHOLD = 0.5


def log_totals(path):
    rows = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        events = r["trace"]["events"]
        rows[(r["case_id"], r["setup"])] = {
            "input_tokens": sum(e.get("input_tokens", 0) for e in events if e["kind"] == "llm_call"),
            "output_tokens": sum(e.get("output_tokens", 0) for e in events if e["kind"] == "llm_call"),
            "elapsed_s": round(sum(e.get("elapsed_s", 0) for e in events if e["kind"] == "workflow_end"), 4),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.MultiIndex.from_tuples(out.index, names=KEY)
    return out


def prepare(raw_dir, extra):
    baseline = pd.read_csv(BASELINE)
    scored = pd.read_csv(raw_dir / "scored.csv")
    if scored.duplicated(KEY).any():
        raise ValueError(f"{raw_dir}: duplicate case/setup rows in scored.csv; deduplicate first")

    out = baseline[KEY + ["category"]].merge(scored, on=KEY, how="left", validate="one_to_one")
    missing = out[out["ticket_id"].isna()]
    if len(missing):
        raise ValueError(f"{raw_dir}: {len(missing)} case/setup rows missing, e.g. {missing[KEY].head(3).values.tolist()}")
    if (out["ticket_id"] != baseline["ticket_id"]).any():
        raise ValueError(f"{raw_dir}: ticket IDs differ from the baseline")

    for m in METRICS:
        if out[f"{m}_score"].isna().any():
            raise ValueError(f"{raw_dir}: missing {m}_score")
        out[f"{m}_success"] = out[f"{m}_success"].astype(str).str.lower().eq("true")
        out[f"{m}_success_manual"] = out[f"{m}_score"] >= THRESHOLD

    totals = log_totals(raw_dir / "results.jsonl")
    out = out.join(totals, on=KEY, validate="one_to_one")
    if out[["input_tokens", "output_tokens", "elapsed_s"]].isna().any().any():
        raise ValueError(f"{raw_dir}: some scored rows have no log in results.jsonl")

    columns = list(baseline.columns)          # same columns, same order as the baseline
    out = out[columns]
    for name, value in extra.items():
        out[name] = value
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--variant", help="value for a 'variant' column (robustness run: paraphrase)")
    p.add_argument("--run-id", help="value for a 'run_id' column (repeated baseline runs: run_2, run_3)")
    a = p.parse_args()
    extra = {k: v for k, v in {"variant": a.variant, "run_id": a.run_id}.items() if v}
    out = prepare(a.raw_dir, extra)
    out.to_csv(a.output, index=False)
    print(f"{a.output}: {len(out)} rows, {len(out.columns)} columns")
    for m in METRICS:
        print(f"  {m}: {out[f'{m}_success_manual'].mean():.1%} pass (score >= {THRESHOLD})")


if __name__ == "__main__":
    main()
