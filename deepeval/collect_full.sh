#!/usr/bin/env bash
# Collect all 13 evaluation cases across all 24 variants; scoring is separate.
# Usage: bash deepeval/collect_full.sh [--variant baseline|paraphrased] [--dry-run] [--api URL] [--case EVAL-01]
#   --variant picks the question wording (deepeval/cases/<variant>.csv, default baseline)
#   and writes to the next free results/deepeval/data/raw/<variant>/runN/ folder.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="$repo_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  printf 'Project Python not found: %s\n' "$python_bin" >&2
  exit 1
fi

variant="baseline"
forward_args=()
while (($#)); do
  case "$1" in
    --variant) variant="${2:?--variant needs a value}"; shift 2 ;;
    --variant=*) variant="${1#*=}"; shift ;;
    *) forward_args+=("$1"); shift ;;
  esac
done

cases_file="deepeval/cases/$variant.csv"
if [[ ! -f "$cases_file" ]]; then
  printf 'Unknown variant %s: %s not found (use baseline or paraphrased).\n' "$variant" "$cases_file" >&2
  exit 1
fi

setup_args=()
for setup in s{1,2,5,6}-{a,b} s{3,4,7,8}-{a,b,c,d}; do
  setup_args+=(--setup "$setup")
done

# Next free run number, so earlier runs are never appended to.
variant_dir="results/deepeval/data/raw/$variant"
run_number=1
while [[ -e "$variant_dir/run$run_number" ]]; do
  run_number=$((run_number + 1))
done
run_dir="$variant_dir/run$run_number"

printf 'Collecting 13 %s cases across 24 variants (312 scenario runs).\n' "$variant"
printf 'The collector resets case state and upserts seed fixtures; use an evaluation database.\n'
printf 'Cases:   %s\n' "$cases_file"
printf 'Outputs: %s/results.jsonl\n' "$run_dir"
printf 'After collection, score with:\n'
printf '  .venv/bin/python deepeval/evaluate.py --input "%s/results.jsonl" --output "%s/scored.csv" --judge-model openai/gpt-oss-120b\n\n' "$run_dir" "$run_dir"

exec "$python_bin" deepeval/run.py \
  --cases "$cases_file" \
  "${setup_args[@]}" \
  --judge-model openai/gpt-oss-120b \
  --output "$run_dir/results.jsonl" \
  ${forward_args[@]+"${forward_args[@]}"}
