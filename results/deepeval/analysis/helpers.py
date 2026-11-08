"""Shared code for the DeepEval analysis notebooks (4_hypothesis_tests, 5_findings).

Everything that is set-up rather than analysis lives here, so the notebooks can stay short:

- paths and chart style
- load_run1():        the 312 scored runs of baseline run 1, labelled and joined with latency, cost,
                      robustness and consistency (one row per case x setup)
- load_three_runs():  runs 1, 2 and 3 stacked (936 rows), for the consistency section
- paired_test() / test_table():  significance tests on matched pairs
- scorecard(), verdict_box(), show_case(), bar_labels():  presentation helpers
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────────────────────────
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
DEEPEVAL = ROOT / "results/deepeval"
RUN1_SCORED = DEEPEVAL / "data/raw/baseline/run1/scored_dedup.csv"
RUN1_LOG = DEEPEVAL / "data/raw/baseline/run1/results.jsonl"
PREPARED = DEEPEVAL / "data/prepared"
TABLES = DEEPEVAL / "output/tables"
FIG_DIR = DEEPEVAL / "output/figures"
CASES_DIR = ROOT / "deepeval/cases"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Agent model price, US$ per million tokens (openai/gpt-oss-120b, configs/base.yaml)
PRICE_IN, PRICE_OUT = 0.03, 0.17
THRESH = 0.5  # DeepEval's default pass threshold

# ── Chart style ──────────────────────────────────────────────────────────────────────────────────
INK, INK2, GRID, PAPER = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GOOD, CRITICAL, AMBER = "#0ca30c", "#d03b3b", "#fab219"
C_A, C_B = "#2a78d6", "#eb6834"          # default left / right side of a two-way comparison
C_AGENTIC, C_GRAPH = C_A, C_B
CONTROL_PAL = {"agentic": C_AGENTIC, "graph": C_GRAPH}
# One fixed colour per option of each design choice, used in every chart of 5_findings
COLOURS = {"agentic": C_AGENTIC, "graph": C_GRAPH,
           "single": "#6250d6", "multi": "#eda100",
           "plan_execute": "#1baf7a", "react": "#e34948",
           "RAG": "#52514e", "SKILL": "#a8a6a0"}
SEQ = sns.light_palette(C_A, as_cmap=True)

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "lines.linewidth": 2,
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
})
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.precision", 3)

NICE = {"agentic": "Agentic", "graph": "Graph", "single": "Single agent", "multi": "Multi-agent",
        "RAG": "RAG", "SKILL": "Skills", "react": "ReAct", "plan_execute": "Plan-and-execute"}

# ── Metric names ─────────────────────────────────────────────────────────────────────────────────
# The eight DeepEval metrics as they appear in the scored CSV
CORE = ["answer_correctness", "tool_correctness", "argument_correctness",
        "task_completion_trace", "step_efficiency_trace", "trajectory_quality_trace"]
PLAN = ["plan_adherence_trace", "plan_quality_trace"]
METRICS = CORE + PLAN
SHORT = {"answer_correctness": "answer", "tool_correctness": "tool", "argument_correctness": "argument",
         "task_completion_trace": "task completion", "step_efficiency_trace": "step efficiency",
         "trajectory_quality_trace": "trajectory", "plan_adherence_trace": "plan adherence",
         "plan_quality_trace": "plan quality", "status_ok": "ticket status"}
PASS_COLS = [f"{m}_pass" for m in METRICS] + ["status_ok"]
SCORE_COLS = [f"{m}_score" for m in METRICS]


def label(col: str) -> str:
    """'tool_correctness_pass' -> 'tool'."""
    return SHORT[col.removesuffix("_pass").removesuffix("_score")]


pct = lambda s: 100 * s.mean()

# Matched pairs: the columns that must be equal for two runs to count as a pair
MATCH_CONTROL = ["pair_id", "case_id"]                    # agentic <-> graph twin (s1-a <-> s5-a), same case
MATCH_AGENTS = ["control", "policy", "variant", "case_id"]  # single <-> multi-agent twin (S1 <-> S3), variants a/b only
MATCH_STRATEGY = ["family", "supervisor", "case_id"]      # react <-> plan worker, supervisor held fixed


# ── Loading ──────────────────────────────────────────────────────────────────────────────────────
def _add_setup_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the design choices from the setup ID, e.g. 's3-c'."""
    fam = df["setup"].str[1].astype(int)
    df["family"] = "S" + fam.astype(str)                                   # S1..S8
    df["variant"] = df["setup"].str[-1]                                    # a..d
    df["control"] = np.where(fam <= 4, "agentic", "graph")
    df["agents"] = np.where(fam.isin([1, 2, 5, 6]), "single", "multi")
    df["policy"] = np.where(fam % 2 == 1, "RAG", "SKILL")
    single = {"a": "react", "b": "plan_execute"}
    multi = {"a": "react / react", "b": "plan / plan", "c": "react / plan", "d": "plan / react"}
    df["strategy"] = np.where(df["agents"] == "single", df["variant"].map(single), df["variant"].map(multi))
    # Thinking mode of whoever calls the tools (the agent, or the specialists in a multi-agent setup)
    df["worker_strategy"] = df["variant"].map({"a": "react", "b": "plan_execute", "c": "plan_execute", "d": "react"})
    # Supervisor's thinking mode in a multi-agent setup; "none" for single agents
    df["supervisor"] = np.where(df["agents"] == "single", "none",
                                df["variant"].map({"a": "react", "b": "plan_execute", "c": "react", "d": "plan_execute"}))
    df["ab_strategy"] = df["variant"].map({"a": "react", "b": "plan_execute"})  # variants a vs b only
    df["pair_id"] = df["agents"] + "-" + df["policy"] + "-" + df["variant"]
    df["arch"] = df["agents"] + " · " + df["control"]
    return df


def _summarise_log_row(row: dict) -> dict:
    """Latency, LLM calls, tokens and tools for one raw run (multi-turn cases are summed over turns)."""
    events = row["trace"].get("events", [])
    ends = [e for e in events if e["kind"] == "workflow_end"]
    calls = [e for e in events if e["kind"] == "llm_call"]
    return {
        "case_id": row["case_id"], "setup": row["setup"], "raw_status": row["ticket_status"],
        "elapsed_s": sum(e.get("elapsed_s", 0) for e in ends),
        "llm_calls": sum(e.get("llm_calls", 0) for e in ends),
        "input_tokens": sum(e.get("input_tokens", 0) for e in calls),
        "output_tokens": sum(e.get("output_tokens", 0) for e in calls),
        "tool_calls": len(row["tools_called"]),
        "tool_path": " > ".join(t["name"] for t in row["tools_called"]),
        "write_count": len(row.get("writes") or []),
        "models": ",".join(sorted({e.get("model", "?") for e in calls})),
    }


def _to_bool(col: pd.Series) -> pd.Series:
    """True/False however it was written (Excel saves TRUE/FALSE); anything else is a data error."""
    out = col.map(lambda v: v if isinstance(v, bool) or pd.isna(v)
                  else {"true": True, "false": False}.get(str(v).strip().lower(), "BAD"))
    if out.eq("BAD").any():
        raise ValueError(f"{col.name}: {out.eq('BAD').sum()} labels are not True/False")
    return out


def load_run1(with_reliability: bool = True) -> pd.DataFrame:
    """The 312 scored runs of baseline run 1, one row per case x setup, ready to analyse.

    Adds: design-choice labels; latency/tokens/cost from the raw log; case tags (kind, category, ...);
    a 0/1 `<metric>_pass` column per DeepEval metric (manual label, else score >= 0.5); `status_ok`;
    our three quality metrics (task_pass, tool_call_pass, step_pass); and, if with_reliability,
    robustness (`*_robust_pass`) and consistency (`*_consistent`) flags from output/tables/.
    """
    df = _add_setup_labels(pd.read_csv(RUN1_SCORED))

    # Latency and cost. The log also holds an earlier EVAL-01 pass per setup; the scored run is the last one.
    raw = pd.DataFrame([_summarise_log_row(json.loads(line)) for line in RUN1_LOG.open(encoding="utf-8")])
    raw = raw.drop_duplicates(["case_id", "setup"], keep="last")
    df = df.merge(raw, on=["case_id", "setup"], how="left", validate="one_to_one")
    assert df["elapsed_s"].notna().all(), "a scored row has no raw run"
    assert (df["raw_status"] == df["ticket_status"]).all(), "raw and scored ticket statuses disagree"
    df["total_tokens"] = df["input_tokens"] + df["output_tokens"]
    df["cost_usd"] = (df["input_tokens"] * PRICE_IN + df["output_tokens"] * PRICE_OUT) / 1e6

    # Case tags
    cases = pd.read_csv(CASES_DIR / "baseline.csv").set_index("case_id")[["focus"]]
    tags = pd.read_csv(CASES_DIR / "categories.csv", index_col="case_id").drop(columns=["focus", "query"])
    tags = tags.rename(columns=lambda c: c.replace("_", " "))
    df = df.join(cases, on="case_id").join(tags, on="case_id")
    df["kind"] = np.where(df["case_id"].isin([f"EVAL-0{i}" for i in range(1, 9)]), "routine lookup", "non-routine")

    # Pass/fail per DeepEval metric: the manual label (score >= 0.5 plus two hand corrections), NaN if not scored
    for m in METRICS:
        has_score = df[f"{m}_score"].notna()
        manual = _to_bool(df[f"{m}_success_manual"])
        rule = (df[f"{m}_score"] >= THRESH).where(has_score)
        df[f"{m}_pass"] = manual.where(manual.notna(), rule).where(has_score).astype(float)
    df["status_ok"] = (df["ticket_status"] == df["expected_ticket_status"]).astype(float)

    # Our quality metrics (1-3)
    df["task_pass"] = df["answer_correctness_pass"].astype(bool) & df["status_ok"].astype(bool)
    df["tool_call_pass"] = df["tool_correctness_pass"].astype(bool) & df["argument_correctness_pass"].astype(bool)
    df["step_pass"] = df["step_efficiency_trace_pass"].astype(bool)

    if with_reliability:
        key = pd.MultiIndex.from_frame(df[["case_id", "setup"]])
        rob = pd.read_csv(TABLES / "robustness_by_setup_and_case.csv").set_index(["case_id", "setup"])
        for c in ["task_robust_pass", "tool_robust_pass", "step_robust_pass"]:
            df[c] = rob[c].reindex(key).astype(bool).values
        cons = pd.read_csv(TABLES / "consistency_by_setup_and_case.csv").set_index(["case_id", "setup"])
        for m in ["task", "tool", "step"]:
            df[f"{m}_consistent"] = cons[f"{m}_same_outcome"].reindex(key).astype(bool).values
    return df


def load_three_runs(run1: pd.DataFrame) -> pd.DataFrame:
    """Runs 1, 2 and 3 stacked (936 rows), with the same labels and pass rules."""
    def flags(frame, run):
        out = frame[["case_id", "setup"]].copy()
        out["run"] = run
        out["status_ok"] = (frame.ticket_status == frame.expected_ticket_status).astype(float).values
        ok = out.status_ok.astype(bool).values
        out["task_pass"] = frame.answer_correctness_success_manual.astype(bool).values & ok
        out["tool_correctness_pass"] = frame.tool_correctness_success_manual.astype(bool).values
        out["tool_call_pass"] = out.tool_correctness_pass.values & frame.argument_correctness_success_manual.astype(bool).values
        out["step_pass"] = frame.step_efficiency_trace_success_manual.astype(bool).values
        out["elapsed_s"] = frame.elapsed_s.values
        out["cost_usd"] = ((frame.input_tokens * PRICE_IN + frame.output_tokens * PRICE_OUT) / 1e6).values
        return out

    reruns = [flags(pd.read_csv(PREPARED / f"baseline_run{i}.csv"), f"run {i}") for i in (2, 3)]
    three = pd.concat([flags(run1, "run 1"), *reruns], ignore_index=True)
    labels = ["setup", "family", "variant", "control", "agents", "policy", "strategy", "worker_strategy", "supervisor", "pair_id"]
    three = (three.merge(run1[labels].drop_duplicates("setup"), on="setup")
                  .merge(run1[["case_id", "kind", "category"]].drop_duplicates("case_id"), on="case_id"))
    assert len(three) == 3 * len(run1), "runs 2 and 3 need the same 312 runs"
    return three


def case_tags() -> pd.DataFrame:
    """The task tags per case (category, complexity, turns, ...), from deepeval/cases/categories.csv."""
    tags = pd.read_csv(CASES_DIR / "categories.csv", index_col="case_id").drop(columns=["focus", "query"])
    return tags.rename(columns=lambda c: c.replace("_", " "))


def label_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two audit tables for the pass/fail labels.

    1. How often the judge's own `_success` flag disagrees with its score (why we relabel).
    2. How often the manual label differs from the score >= 0.5 rule (the hand corrections).
    """
    flags = pd.DataFrame({SHORT[m]: {
        "score ≥ 0.5 but flagged False": int(((df[f"{m}_score"] >= THRESH) & (df[f"{m}_success"].astype(str) == "False")).sum()),
        "score < 0.5 but flagged True": int(((df[f"{m}_score"] < THRESH) & (df[f"{m}_success"].astype(str) == "True")).sum()),
    } for m in METRICS}).T
    overrides = {}
    for m in METRICS:
        has_score = df[f"{m}_score"].notna()
        manual = _to_bool(df[f"{m}_success_manual"])
        rule = (df[f"{m}_score"] >= THRESH).where(has_score)
        both = manual.notna() & has_score
        overrides[SHORT[m]] = {
            "manual pass, score < 0.5": int((both & manual.eq(True) & rule.eq(False)).sum()),
            "manual fail, score ≥ 0.5": int((both & manual.eq(False) & rule.eq(True)).sum()),
            "no manual label → score rule": int((manual.isna() & has_score).sum()),
        }
    return flags, pd.DataFrame(overrides).T


def compact_style() -> None:
    """Smaller fonts, for the dense heatmaps in 4_hypothesis_tests."""
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 9, "figure.dpi": 110, "axes.edgecolor": GRID})


_TRANSCRIPTS: dict | None = None


def run_of(case_id: str, setup: str) -> dict:
    """The raw run-1 log entry (transcript and trace) for one case x setup."""
    global _TRANSCRIPTS
    if _TRANSCRIPTS is None:
        _TRANSCRIPTS = {}
        for line in RUN1_LOG.open(encoding="utf-8"):
            r = json.loads(line)
            _TRANSCRIPTS[(r["case_id"], r["setup"])] = r   # later lines overwrite: the scored run is kept
    return _TRANSCRIPTS[(case_id, setup)]


# ── Evidence from the raw traces ─────────────────────────────────────────────────────────────────
WRITE_TOOLS = {"cancel_order", "request_refund", "create_return_request", "backend_write"}


def write_gate_refusals(case_id: str, setup: str, control: str) -> dict:
    """How often the write gate refused a write in one run, and how many write proposals the run made.

    Graph runs log a 'proposal_refused' event and then restart gather -> propose ('proposal' events).
    Agentic runs get the refusal back as the write tool's result (an error, 'blocked' or 'review_required').
    """
    ev = run_of(case_id, setup)["trace"]["events"]
    if control == "graph":
        refusals = sum(e["kind"] == "proposal_refused" for e in ev)
    else:
        refusals = sum(e["kind"] == "tool_end" and e.get("tool") in WRITE_TOOLS and isinstance(e.get("output"), dict)
                       and ("error" in e["output"] or e["output"].get("status") in ("blocked", "review_required"))
                       for e in ev)
    return {"refusals": refusals, "proposal_cycles": sum(e["kind"] == "proposal" for e in ev)}


def time_after_first_refusal(case_id: str, setup: str) -> dict:
    """Seconds of a run spent before vs after the write gate's first refusal of a graph proposal.

    Everything after the first refusal is the graph restarting gather -> propose. Multi-turn cases are summed
    over turns. Agentic runs never log 'proposal_refused', so all their time counts as 'before'.
    """
    before = after = 0.0
    start = refused = None
    for e in run_of(case_id, setup)["trace"]["events"]:
        t = float(e["t"])
        if e["kind"] == "workflow_start":
            start, refused = t, None
        elif e["kind"] == "proposal_refused" and refused is None:
            refused = t
        elif e["kind"] == "workflow_end":
            cut = refused if refused is not None else t
            before += cut - start
            after += t - cut
    return {"s_before_refusal": before, "s_after_refusal": after}


def first_refusal_reason(case_id: str, setup: str) -> str | None:
    """The reason text of the first write-gate refusal in a graph run."""
    return next((e["reason"] for e in run_of(case_id, setup)["trace"]["events"] if e["kind"] == "proposal_refused"), None)


def tools_vs_expected(case_id: str, setup: str) -> dict:
    """Tools called in a run vs the gold case's expected tools (calls to tools not in the gold set = over-fetching)."""
    r = run_of(case_id, setup)
    expected = {t["name"] for t in r["expected_tools"]}
    called = [t["name"] for t in r["tools_called"]]
    return {"tools called": len(called), "expected tools": len(r["expected_tools"]),
            "calls beyond expected": sum(n not in expected for n in called)}


def refund_reply_check(case_id: str, setup: str) -> dict:
    """For the refund case: was the refund executed, and does the reply promise a payout timeline?"""
    import re
    r = run_of(case_id, setup)
    return {"refund executed": any(w.get("status") == "executed" and w.get("operation") == "refund"
                                   for w in (r.get("writes") or [])),
            "reply promises a timeline": bool(re.search(r"business day|working day|within \d", r["actual_output"], re.I))}


def pareto_front(data: pd.DataFrame, cost_col: str, gain_col: str) -> pd.DataFrame:
    """Rows not beaten by any other row (no other row is both cheaper and better), sorted by cost."""
    best, keep = -np.inf, []
    for i, r in data.sort_values([cost_col, gain_col], ascending=[True, False]).iterrows():
        if r[gain_col] > best:
            keep.append(i)
            best = r[gain_col]
    return data.loc[keep]


# ── Significance tests on matched pairs ──────────────────────────────────────────────────────────
def paired_test(data: pd.DataFrame, metric: str, *, compare: str, a: str, b: str, match_on: list[str],
                name: str | None = None, stat: str = "median") -> dict:
    """One significance test, a vs b, on runs that are identical except for `compare`.

    Pass/fail metrics (0/1) use the exact McNemar test (only pairs where one side passes count);
    numeric metrics (latency, calls) use the Wilcoxon signed-rank test on the paired differences.
    """
    w = data.assign(_v=data[metric].astype(float)).pivot_table(
        index=match_on, columns=compare, values="_v", aggfunc="first").dropna()
    binary = set(np.unique(w.values)) <= {0.0, 1.0}
    row = {"Comparison": name or metric, "Test": "McNemar" if binary else "Wilcoxon", "Pairs": len(w)}
    if binary:
        only_a = int(((w[a] == 1) & (w[b] == 0)).sum())
        only_b = int(((w[a] == 0) & (w[b] == 1)).sum())
        p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        row |= {"A": 100 * w[a].mean(), "B": 100 * w[b].mean(), "Unit": "% pass",
                "Only A passes": only_a, "Only B passes": only_b}
    else:
        f = np.median if stat == "median" else np.mean
        d = w[a] - w[b]
        p = stats.wilcoxon(w[a], w[b], zero_method="zsplit").pvalue if (d != 0).any() else 1.0
        row |= {"A": f(w[a]), "B": f(w[b]), "Unit": stat, "Only A passes": np.nan, "Only B passes": np.nan}
    row |= {"p": p, "Significant": "yes" if p < 0.05 else "no"}
    return row


def test_table(rows: list[dict], a: str, b: str, show: bool = True) -> pd.DataFrame:
    """Collect paired_test() rows into one table (indexed by comparison name) and display it."""
    t = pd.DataFrame(rows).set_index("Comparison")
    if show:
        display(t.style.format({"A": "{:.1f}", "B": "{:.1f}", "Only A passes": "{:.0f}",
                                "Only B passes": "{:.0f}", "p": "{:.3g}"}, na_rep="–")
                .map(lambda v: f"font-weight:bold;color:{GOOD}" if v == "yes" else "", subset=["Significant"])
                .set_caption(f"<b>Matched-pair tests</b>: A = {NICE.get(a, a)}, B = {NICE.get(b, b)}"))
    return t



def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values, in the same order as the input (controls the family-wise error rate)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out

# ── Presentation helpers ─────────────────────────────────────────────────────────────────────────
SCORECARD_ROWS = ["1 Task success (%)", "2 Tool-call accuracy (%)", "3 Step efficiency (%)", "4 Cost (median US$)",
                  "5 Latency (median s)", "6 Robustness (%)", "7 Consistency (%)"]


def scorecard(df: pd.DataFrame, compare: str, a: str, b: str, show: bool = True) -> pd.DataFrame:
    """The seven project metrics for both sides of one design choice (all runs pooled)."""
    def col(d):
        return [pct(d.task_pass), pct(d.tool_call_pass), pct(d.step_pass), d.cost_usd.median(),
                d.elapsed_s.median(), pct(d.task_robust_pass), pct(d.task_consistent)]
    A, B = df[df[compare] == a], df[df[compare] == b]
    card = pd.DataFrame({NICE[a]: col(A), NICE[b]: col(B)}, index=SCORECARD_ROWS)
    card["Difference"] = card[NICE[a]] - card[NICE[b]]
    if show:
        num = list(card.columns)
        display(card.style.format("{:.1f}", subset=num)
                .format("{:.5f}", subset=pd.IndexSlice[["4 Cost (median US$)"], num])
                .set_caption(f"<b>{NICE[a]} vs {NICE[b]}</b> ({len(A)} vs {len(B)} runs)"))
    return card


def verdict_box(code: str, claim: str, verdict: str, points: list[str]) -> None:
    """Coloured verdict banner: green = supported, amber = partly, red = not supported."""
    colour = GOOD if verdict.lower().startswith("supported") else CRITICAL if verdict.lower().startswith("not") else AMBER
    items = "".join(f"<li>{p}</li>" for p in points)
    display(HTML(
        f'<div style="background:{PAPER};color:{INK};border:1px solid {GRID};border-left:8px solid {colour};'
        f'border-radius:8px;padding:12px 18px;margin:6px 0;font-size:15px;line-height:1.45">'
        f'<div style="font-size:18px"><b>{code}</b> · <i>{claim}</i></div>'
        f'<div style="margin:6px 0"><span style="background:{colour};color:{INK if colour == AMBER else "white"};'
        f'padding:3px 10px;border-radius:12px;font-weight:700">{verdict}</span></div>'
        f'<ul style="margin:4px 0 0 0">{items}</ul></div>'))


def show_case(case_id: str, setup: str, note: str = "") -> None:
    """One real run as a card: customer message, expected outcome, tools, ticket status and reply."""
    r = run_of(case_id, setup)
    ok = r["ticket_status"] == r["expected_ticket_status"]
    status = (f'<span style="color:{GOOD if ok else CRITICAL};font-weight:700">{"✓" if ok else "✗"} '
              f'{r["ticket_status"]}</span> (expected {r["expected_ticket_status"]})')
    tools = ", ".join(t["name"] for t in r["tools_called"]) or "none"
    said = r["input"] if isinstance(r["input"], str) else " / ".join(map(str, r["input"]))
    display(HTML(
        f'<div style="background:{PAPER};color:{INK};border:1px solid {GRID};border-radius:10px;'
        f'padding:14px 18px;margin:8px 0;font-size:15px;line-height:1.45">'
        f'<div style="font-size:17px;font-weight:700">{case_id} · {setup} {note}</div>'
        f'<div><b>Customer:</b> {said}</div>'
        f'<div style="color:{INK2}"><b>Policy expects:</b> {r["ground_truth"]}</div>'
        f'<div><b>Tools called:</b> {tools} &nbsp;|&nbsp; <b>Ticket:</b> {status}</div>'
        f'<div style="margin-top:8px;padding:10px;background:white;border-left:4px solid {C_A}">'
        f'{r["actual_output"].replace(chr(10), "<br>")}</div></div>'))


def bar_labels(ax, bars, fmt=lambda v: f"{v:.0f}%", pad=1.5, size=12) -> None:
    """Write each bar's value above it."""
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + pad, fmt(v), ha="center", fontsize=size,
                fontweight="bold", color=INK)


def paired_bars(ax, labels, a_vals, b_vals, a_name, b_name, fmt=lambda v: f"{v:.0f}%", pad=1.5,
                colours=(C_A, C_B)) -> None:
    """Side-by-side bars for a two-way comparison, with value labels."""
    x = np.arange(len(labels))
    for off, vals, name, colour in [(-0.2, a_vals, a_name, colours[0]), (0.2, b_vals, b_name, colours[1])]:
        bar_labels(ax, ax.bar(x + off, vals, width=0.38, color=colour, label=name), fmt, pad)
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)


def save(fig, name: str) -> None:
    """Save a figure to output/figures/ (so it can go on a slide) and show it."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / name)
    plt.show()
