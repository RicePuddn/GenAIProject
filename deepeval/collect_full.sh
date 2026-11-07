#!/usr/bin/env bash
# Collect all 13 evaluation cases across all 24 variants; scoring is separate.
# Usage: bash deepeval/collect_full.sh [--dry-run] [--api URL] [--case EVAL-01]
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="$repo_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  printf 'Project Python not found: %s\n' "$python_bin" >&2
  exit 1
fi

setup_args=()
for setup in s{1,2,5,6}-{a,b} s{3,4,7,8}-{a,b,c,d}; do
  setup_args+=(--setup "$setup")
done

run_dir="results/deepeval/rerun_$(date +%Y%m%d_%H%M%S)_$$"

printf 'Collecting 13 cases across 24 variants (312 scenario runs).\n'
printf 'The collector resets case state and upserts seed fixtures; use an evaluation database.\n'
printf 'Outputs: %s/results.jsonl\n' "$run_dir"
printf 'After collection, score with:\n'
printf '  .venv/bin/python deepeval/evaluate.py --input "%s/results.jsonl" --output "%s/scored.csv" --judge-model openai/gpt-oss-120b\n\n' "$run_dir" "$run_dir"

exec "$python_bin" deepeval/run.py \
  --cases deepeval/cases_full.csv \
  "${setup_args[@]}" \
  --judge-model openai/gpt-oss-120b \
  --output "$run_dir/results.jsonl" \
  "$@"
