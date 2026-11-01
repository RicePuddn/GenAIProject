"""Demo page for every setup: chat as the customer and watch each agent step live.

An ordinary client of the stateless API: it keeps the conversation and sends the earlier turns as
`history`, and reads /chat/stream so each step (guard, plans, tool calls, proposals, reviews, writes,
handoffs) appears while the turn runs. Start the API first, then from the repo root:

    .venv/bin/uvicorn support_ops.api:app --host 127.0.0.1 --port 8000
    .venv/bin/streamlit run apps/demo.py

Set SUPPORT_OPS_API to use an API somewhere other than http://127.0.0.1:8000.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Iterator

import streamlit as st

API = os.environ.get("SUPPORT_OPS_API", "http://127.0.0.1:8000").rstrip("/")
STATUS_LABELS = {
    "resolved": ("success", "Resolved"),
    "awaiting_customer": ("info", "Waiting for the customer"),
    "awaiting_human": ("warning", "Awaiting a human operator"),
    "in_progress": ("error", "Cut off: the LLM provider failed. Send the message again"),
}
NODE_TITLES = {"gather": "Gather facts", "propose": "Propose", "check": "Rule check", "write_review": "Write review",
               "reply": "Reply and reply check"}


def open_api(path: str, payload: dict[str, Any] | None, timeout: float):
    request = urllib.request.Request(
        API + path, method="POST" if payload is not None else "GET",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"API error {exc.code}: {exc.read().decode(errors='replace')}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"cannot reach the API at {API} ({exc.reason}). Is uvicorn running? In RAG mode it accepts "
            "requests only after 'Application startup complete' (about 20 s of model loading); then refresh."
        ) from exc


def call_api(path: str, timeout: float = 10) -> Any:
    with open_api(path, None, timeout) as response:
        return json.loads(response.read())


def stream_chat(payload: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Read Server-Sent Events from /chat/stream as (event, data) pairs, as they arrive."""
    with open_api("/chat/stream", payload, timeout=600) as response:
        event, data = None, []
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].strip())
            elif not line and event:
                yield event, json.loads("\n".join(data))
                event, data = None, []


def new_ticket() -> None:
    st.session_state.update(ticket_id=None, history=[], exchanges=[])


def setup_label(setup_id: str, setup: dict[str, Any]) -> str:
    strategy = ", ".join(f"{role} {name}" for role, name in setup["strategy"].items())
    agents = "single agent" if setup["agents"] == "single" else "multi-agent"
    return f"{setup['name']}-{setup['variant']} · {agents}, {setup['control']}, {setup['policy_mode']} · {strategy}"


# ---------------------------------------------------------------- timeline


def show_tool_result(tool: str, output: dict[str, Any]) -> None:
    if "error" in output:
        st.error(output["error"])
    elif tool == "policy" and "clauses" in output:
        st.caption(f"Retrieved for query: “{output.get('query', '')}”")
        st.dataframe(
            [{"clause": c["id"], "title": c.get("title"), "score": c.get("score"), "via": c.get("via"),
              "text": c.get("text")} for c in output["clauses"]],
            hide_index=True, width="stretch",
        )
    else:
        st.json(output, expanded=False)


def show_event(event: dict[str, Any]) -> None:
    """One trace event as a timeline step; used both live and when redrawing earlier turns."""
    kind = event["kind"]
    if kind == "guard_classification":
        score = event.get("malicious_score")
        st.markdown(f"**Guard:** {event['label']}" + (f" (malicious score {score:.4f})" if score is not None else ""))
    elif kind == "agent_start":
        st.markdown(f"#### {event['agent'].replace('_', ' ').capitalize()} ({event['strategy']})")
    elif kind == "node_start" and event.get("node") in NODE_TITLES:
        st.markdown(f"**{NODE_TITLES[event['node']]}**" + (f" · {event['agent'].replace('_', ' ')}"
                                                          if event.get("agent") else ""))
    elif kind == "llm_response" and event.get("reasoning"):
        st.markdown(f"**Thought ({event['role']})**")
        st.info(event["reasoning"])
    elif kind == "plan":
        label = "Plan (completion check)" if event.get("check") else "Plan"
        st.markdown(f"**{label}:** " + (" → ".join(f"`{name}`" for name, _ in event["calls"])
                                        if event["calls"] else "nothing more to do"))
    elif kind == "plan_rejected":
        st.warning(f"Planning again ({event['replan']}): {event['reason']}")
    elif kind == "tool_call_requested":
        st.markdown(f"**Tool:** `{event['tool']}` `{json.dumps(event['input'])}`")
    elif kind == "tool_end":
        st.caption(f"`{event['tool']}` finished in {event['elapsed_s']:.2f}s")
        show_tool_result(event["tool"], event["output"])
    elif kind == "control_call" and event["tool"] == "ask_specialist":
        st.markdown(f"**Supervisor asks {event['input'].get('specialist', '').replace('_', ' ')}:** "
                    f"{event['input'].get('request', '')}")
    elif kind == "control_call" and event["tool"] == "finish":
        st.markdown(f"**Finish:** {event['input'].get('outcome')} · {event['input'].get('note')}")
    elif kind == "control_result" and event["tool"] == "ask_specialist" and "outcome" in event["output"]:
        st.caption(f"{event['output']['specialist'].replace('_', ' ')} ended {event['output']['outcome']}: "
                   f"{event['output'].get('note') or ''}")
    elif kind == "proposal":
        writes = "; ".join(f"{w['operation']} {w['order_id']}" + (f" {w['amount']}" if w.get("amount") else "")
                           for w in event.get("writes", []))
        st.markdown(f"**Proposal:** {event['outcome']} · {event['note']}" + (f" · writes: {writes}" if writes else ""))
    elif kind == "proposal_refused":
        st.warning(f"Back to gathering: {event['reason']}")
    elif kind == "approval":
        text = f"**Approver ({event['stage']}):** {event['decision']} — {event['reason']}"
        if event["decision"] == "approve":
            st.success(text)
        else:
            st.warning(text + "".join(f"\n- {issue}" for issue in event["issues"]))
    elif kind == "write_result":
        result = event["result"]
        amount = f" {result['amount']} {result.get('currency', '')}" if result.get("amount") else ""
        st.success(f"Recorded {result['operation']}{amount} on {result['order_id']} ({result['write_id']})")
    elif kind == "reply_wording_rejected":
        st.warning("Reply redrafted: " + "; ".join(event["issues"]))
    elif kind == "repeated_error":
        st.error(f"`{event['tool']}` failed {event['times']} times in a row with the same error; the agent stopped")
    elif kind == "tool_retry":
        st.caption(f"`{event['tool']}` failed ({event['error'][:150]}), running it again")
    elif kind in {"json_retry", "json_failed"}:
        st.caption(f"Malformed reply from the {event['role']} (attempt {event['attempt']}): {event['error'][:200]}")
    elif kind == "turn_stopped":
        st.warning(f"Stopped for a person: {event['reason']}")
    elif kind == "budget_exceeded":
        st.error(f"Budget reached: {event['reason']}")
    elif kind == "handoff_created":
        st.warning(f"Handed off to a human: {event['packet'].get('reason')}")


def summary(response: dict[str, Any]) -> str:
    trace = response["trace"]
    strategy = ", ".join(f"{role} {name}" for role, name in response["strategy"].items())
    return (f"What happened · {response['setup']} · {strategy} · {trace['tool_call_count']} tools · "
            f"{trace['llm_calls']} LLM calls · ${trace['cost_usd']:.4f} · {trace['elapsed_s']:.1f}s")


def show_exchange(exchange: dict[str, Any], expanded: bool = False) -> None:
    with st.chat_message("user"):
        st.markdown(exchange["message"])
    with st.chat_message("assistant"):
        if "error" in exchange:
            st.error(exchange["error"])
            return
        response = exchange["response"]
        st.markdown(response["reply"])
        kind, label = STATUS_LABELS.get(response["ticket_status"], ("info", response["ticket_status"]))
        getattr(st, kind)(label)
        trace = response["trace"]
        with st.expander(summary(response), expanded=expanded):
            columns = st.columns(4)
            columns[0].metric("LLM calls", trace["llm_calls"])
            columns[1].metric("Tokens in / out", f"{trace['input_tokens']} / {trace['output_tokens']}")
            columns[2].metric("Cost (USD)", f"{trace['cost_usd']:.4f}")
            columns[3].metric("Time (s)", f"{trace['elapsed_s']:.1f}")
            for event in trace["events"]:
                show_event(event)
            if response.get("parts"):
                st.markdown("**Parts**")
                st.dataframe([{key: part.get(key) for key in ("specialist", "request", "outcome", "note")}
                              for part in response["parts"]], hide_index=True, width="stretch")
            if response.get("handoff_packet"):
                st.markdown("**Handoff packet**")
                st.json(response["handoff_packet"], expanded=False)


# ---------------------------------------------------------------- page

st.set_page_config(page_title="E-Commerce Support", layout="wide")
if "history" not in st.session_state:
    new_ticket()

with st.sidebar:
    st.header("E-Commerce Support")
    try:
        info = call_api("/info", timeout=10)
    except RuntimeError as exc:
        info = {"default_setup": None, "setups": {}, "clock_start": None, "models": {}}
        st.error(str(exc))
    setups = info["setups"]
    # Switching setup or account starts a new ticket, so one conversation never spans two of either.
    setup_id = st.selectbox("Setup", list(setups), key="setup_id", on_change=new_ticket,
                            index=list(setups).index(info["default_setup"]) if info["default_setup"] in setups else 0,
                            format_func=lambda key: setup_label(key, setups[key]))
    try:
        accounts = call_api("/customers", timeout=10)
    except RuntimeError as exc:
        accounts = []
        st.warning(f"Accounts unavailable, chatting signed out: {exc}")
    names = {row["id"]: f"{row['first_name']} {row['last_name']} ({row['id']})" for row in accounts}
    customer_id = st.selectbox("Signed in as", [None, *names], key="customer_id", on_change=new_ticket,
                               format_func=lambda account: names.get(account, "Not signed in"),
                               help="The agents see and change only this customer's orders, and the "
                                    "ticket is saved as theirs. Signed out, nothing is scoped.")
    today = st.toggle("Use today's date", key="today",
                      help=f"Off: the workflow date is run.clock_start ({info['clock_start']}), as in experiment "
                           "runs. On: today, so orders delivered in the last 30 days are within their return window.")
    st.button("New ticket", on_click=new_ticket, width="stretch")
    st.caption(f"Ticket: {st.session_state.ticket_id or 'new'} · {len(st.session_state.history)} turns sent as history")
    if info["models"]:
        st.caption(f"Model: {info['models'].get('agent')} · effort {info.get('reasoning_effort')}")

exchanges = st.session_state.exchanges
for index, exchange in enumerate(exchanges):
    show_exchange(exchange, expanded=index == len(exchanges) - 1)

if message := st.chat_input("Write as the customer, e.g. Can I return order ORD-2026-00041?"):
    exchange: dict[str, Any] = {"message": message}
    payload: dict[str, Any] = {"customer_message": message, "history": st.session_state.history}
    if setup_id:
        payload["setup"] = setup_id
    if st.session_state.ticket_id:
        payload["ticket_id"] = st.session_state.ticket_id
    if customer_id:
        payload["customer_id"] = customer_id
    if today:
        payload["clock_start"] = datetime.now(timezone.utc).isoformat()
    with st.chat_message("user"):
        st.markdown(message)
    with st.chat_message("assistant"):
        # Steps are drawn as their events arrive; the page is redrawn in full once the turn ends.
        with st.status("Working…", expanded=True) as status:
            try:
                for event, data in stream_chat(payload):
                    if event == "trace":
                        show_event(data)
                    elif event == "result":
                        exchange["response"] = data
                    elif event == "error":
                        exchange["error"] = data["detail"]
            except RuntimeError as exc:
                exchange["error"] = str(exc)
            if not ({"response", "error"} & exchange.keys()):
                exchange["error"] = "the stream ended without a result"
            status.update(label="Done" if "response" in exchange else "Failed",
                          state="complete" if "response" in exchange else "error")
    if "response" in exchange:
        st.session_state.ticket_id = exchange["response"]["ticket_id"]
        st.session_state.history = exchange["response"]["history"]
    exchanges.append(exchange)
    st.rerun()
