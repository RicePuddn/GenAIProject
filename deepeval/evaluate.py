#!/usr/bin/env python3
"""Score saved SupportOpsBot run JSONL with DeepEval; does not call the application API."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SupportOpsBot.runtime.shared.config import load_env


def as_tool_call(raw: dict[str, Any]):
    from deepeval.test_case import ToolCall

    return ToolCall(name=raw.get("name", "unknown"),
                    input_parameters=raw.get("arguments") or {},
                    output=raw.get("output"))


def parse_expected(raw: Any, setup: str) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        raw = raw.get(setup, [])
    return raw or []


def trace_text(row: dict[str, Any]) -> str:
    """A compact ordered trace for G-Eval. Keep tool payloads and reasoning for evidence."""
    steps = []
    for event in row.get("trace", {}).get("events", []):
        kind = event.get("kind")
        if kind in {"llm_response", "plan", "plan_created", "tool_call_requested", "tool_end",
                    "control_call", "control_result", "proposal", "write_result", "write_review",
                    "approval", "provider_error", "turn_stopped", "handoff_created", "workflow_end"}:
            item = {k: event[k] for k in ("kind", "role", "agent", "tool", "input", "output", "reasoning",
                                            "calls", "plan", "outcome", "note", "result", "decision", "error",
                                            "reason", "status") if k in event}
            steps.append(item)
    return json.dumps(steps, ensure_ascii=False, default=str)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("deepeval/results.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("deepeval/scored.csv"))
    parser.add_argument("--case", action="append", help="Filter by case ID or ticket ID; repeatable")
    parser.add_argument("--setup", action="append", help="Filter by setup ID; repeatable")
    parser.add_argument("--start-at", metavar="CASE,SETUP,TICKET",
                        help="Resume scoring at this row (inclusive), e.g. EVAL-02,s3-b,TKT-EVAL-03")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge-model", default=os.getenv("DEEPEVAL_JUDGE_MODEL", "openai/gpt-oss-120b"))
    parser.add_argument("--max-output-tokens", type=int, default=1024,
                        help="Cap each judge completion to avoid requesting the model's full context window")
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        parser.error("OPENROUTER_API_KEY is missing; add it to the project .env or environment")
    if not args.input.is_file():
        parser.error(f"input JSONL not found: {args.input}")

    from deepeval.metrics import (ArgumentCorrectnessMetric, GEval, ToolCorrectnessMetric)
    from deepeval.models import OpenRouterModel
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    judge = OpenRouterModel(model=args.judge_model, api_key=api_key,
                            base_url="https://openrouter.ai/api/v1", temperature=0)
    metrics = {
        "answer_correctness": GEval(
            name="Answer Correctness",
            criteria="Assess whether the customer-facing answer accurately and completely answers the user's request, agrees with the ground truth, and does not invent unsupported claims.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT], model=judge),
        "tool_correctness": ToolCorrectnessMetric(include_reason=True, model=judge,
                                                   should_consider_ordering=True),
        "argument_correctness": ArgumentCorrectnessMetric(include_reason=True, model=judge),
        "task_completion_trace": GEval(
            name="Task Completion (Recorded Trace)",
            criteria="Assess from the recorded trace whether the agent completed the requested task safely. A write only counts as completed if a successful write_result exists; a request sent for human review is not completed. A read-only question is completed when the answer is supported by tool results. Treat provider errors, missing finish, or an unresolved request as incomplete.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT], model=judge),
        "step_efficiency_trace": GEval(
            name="Step Efficiency (Recorded Trace)",
            criteria="Assess whether the ordered trace uses only relevant tools and reasoning steps, avoids repeated or unrelated calls, and reaches a supported outcome directly. Penalize hallucinated IDs, repeated failed calls, unnecessary broad reads, and loops.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT], model=judge),
        "trajectory_quality_trace": GEval(
            name="Trajectory Quality (Recorded Trace)",
            criteria="Assess whether the agent's ordered trace uses evidence correctly, follows required policy retrieval and write safeguards, and produces an outcome consistent with tool results. Penalize unsupported conclusions and missing required steps.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT], model=judge),
    }
    plan_metrics = {
        "plan_adherence_trace": GEval(
            name="Plan Adherence (Recorded Trace)",
            criteria="Compare the plan or stated approach in the trace with subsequent tool calls and outcome. Score highly when execution follows the plan, and lower when it skips required steps, changes IDs, or abandons the task.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT], model=judge),
        "plan_quality_trace": GEval(
            name="Plan Quality (Recorded Trace)",
            criteria="Assess whether the plan or stated approach is complete, logically ordered, safe, and sufficient for the customer request. A simple request may need no explicit plan; do not penalize a concise direct approach.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT], model=judge),
    }

    rows = []
    with args.input.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON at line {line_no}: {exc}", file=sys.stderr)
                continue
            if args.case and row.get("case_id") not in args.case and row.get("ticket_id") not in args.case:
                continue
            if args.setup and row.get("setup") not in args.setup:
                continue
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break
    if args.start_at:
        parts = [part.strip() for part in args.start_at.split(",")]
        if len(parts) != 3:
            parser.error("--start-at must be CASE,SETUP,TICKET")
        start_index = next((i for i, row in enumerate(rows)
                            if (row.get("case_id"), row.get("setup"), row.get("ticket_id")) == tuple(parts)), None)
        if start_index is None:
            parser.error(f"resume row not found in input: {args.start_at}")
        rows = rows[start_index:]
    if not rows:
        parser.error("no result rows matched the selected input/filters")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    metric_columns = ["answer_correctness", "tool_correctness", "argument_correctness",
                      "task_completion_trace", "step_efficiency_trace", "trajectory_quality_trace",
                      "plan_adherence_trace", "plan_quality_trace"]
    fieldnames = ["case_id", "setup", "ticket_id", "ticket_status", "expected_ticket_status"]
    fieldnames += [part for name in metric_columns for part in (f"{name}_score", f"{name}_success", f"{name}_reason")]
    with args.output.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            expected_tools = parse_expected(row.get("expected_tools", []), row.get("setup", ""))
            called = row.get("tools_called", [])
            tc = LLMTestCase(
                input=row.get("input", ""), actual_output=row.get("actual_output", ""),
                expected_output=row.get("ground_truth", ""),
                tools_called=[as_tool_call(item) for item in called],
                expected_tools=[as_tool_call(item) for item in expected_tools],
            )
            # Keep the final-answer metrics focused on the customer-facing answer.
            trace_case = LLMTestCase(
                input=row.get("input", ""),
                actual_output=(row.get("actual_output", "") + "\n\nRECORDED EXECUTION TRACE:\n" + trace_text(row)),
                expected_output=row.get("ground_truth", ""),
            )
            row_out = {"case_id": row.get("case_id"), "setup": row.get("setup"),
                       "ticket_id": row.get("ticket_id"), "ticket_status": row.get("ticket_status"),
                       "expected_ticket_status": row.get("expected_ticket_status")}
            selected = {"answer_correctness": metrics["answer_correctness"],
                        "tool_correctness": metrics["tool_correctness"],
                        "argument_correctness": metrics["argument_correctness"]}
            trace_events = row.get("trace", {}).get("events", [])
            uses_plan = any("plan_execute" in str(event.get("strategy", "")) for event in trace_events)
            selected["task_completion_trace"] = metrics["task_completion_trace"]
            selected["step_efficiency_trace"] = metrics["step_efficiency_trace"]
            selected["trajectory_quality_trace"] = metrics["trajectory_quality_trace"]
            if uses_plan:
                selected.update(plan_metrics)
            for name, metric in selected.items():
                try:
                    metric.measure(trace_case if name.endswith("_trace") else tc)
                    row_out[f"{name}_score"] = metric.score
                    row_out[f"{name}_reason"] = metric.reason
                    row_out[f"{name}_success"] = metric.is_successful()
                except Exception as exc:  # Keep one judge/provider failure from discarding other rows.
                    failures += 1
                    row_out[f"{name}_reason"] = f"ERROR: {type(exc).__name__}: {exc}"
            writer.writerow(row_out)
            target.flush()
            print(f"[{index}/{len(rows)}] {row.get('case_id')} {row.get('setup')}: "
                  + ", ".join(f"{name}={row_out.get(f'{name}_score', 'ERR')}" for name in metric_columns
                               if row_out.get(f"{name}_score") not in (None, "")))
    print(f"Wrote scores for {len(rows)} result rows to {args.output}; metric errors: {failures}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
