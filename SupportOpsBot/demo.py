"""Demo page for every setup: chat as the customer and watch each agent step live.

An ordinary client of the stateless API: it keeps the conversation and sends the earlier turns as
`history`, and reads /chat/stream so each step (guard, plans, tool calls, proposals, reviews, writes,
handoffs) appears while the turn runs. Start the API first, then from the repo root:

    .venv/bin/uvicorn SupportOpsBot.chatRoute:app --host 127.0.0.1 --port 8000
    .venv/bin/streamlit run SupportOpsBot/demo.py

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
    "in_progress": ("error", "Request incomplete. Send the message again"),
}
ROLE_TITLES = {
    "agent": "Customer Support Agent",
    "supervisor": "Supervisor Agent",
    "specialist": "Specialist Agent",
    "specialists": "Specialist Agents",
    "orders_shipping": "Orders and Shipping Agent",
    "returns_refunds": "Returns and Refunds Agent",
    "guard": "Prompt guard (LLM)",
    "approver": "Reviewer (LLM)",
    "reply_writer": "Response writer (LLM)",
}
STRATEGY_TITLES = {"react": "ReAct", "plan_execute": "Plan-and-execute"}
NODE_TITLES = {
    "guard": "Prompt guard (LLM) — screen the customer message",
    "gather": "Graph workflow — gather evidence",
    "propose": "Graph workflow — request an action proposal",
    "check": "Write gate — check business rules and policy evidence",
    "write_review": "Reviewer (LLM) — review the proposed write",
    "reply": "Response writer (LLM) — draft the customer response",
}
TOOL_TITLES = {
    "get_customer_orders": "Look up the customer's orders",
    "get_order": "Look up order details",
    "get_order_items": "Look up items in the order",
    "get_shipments": "Look up shipment details",
    "get_tracking_events": "Look up tracking updates",
    "get_payments": "Look up payment records",
    "get_returns_and_refunds": "Look up existing returns and refunds",
    "create_return_request": "Request a return through the write gate",
    "request_refund": "Request a refund through the write gate",
    "cancel_order": "Request cancellation through the write gate",
    "request_review": "Request independent approval for a write",
    "finish": "Submit the outcome and finish the task",
    "ask_specialist": "Delegate a task to a specialist",
    "backend_write": "Submit a proposed write to the graph's write gate",
}


def role_title(name: str) -> str:
    if name in ROLE_TITLES:
        return ROLE_TITLES[name]
    title = name.replace("_", " ").title()
    return title if title.endswith(" Agent") else f"{title} Agent"


def strategy_title(name: str) -> str:
    return STRATEGY_TITLES.get(name, name.replace("_", " ").capitalize())


def status_title(status: str) -> str:
    return STATUS_LABELS.get(status, ("info", status.replace("_", " ")))[1]


def tool_title(tool: str, arguments: dict[str, Any] | None = None) -> str:
    if tool == "policy":
        return {
            "search": "Search for relevant policy clauses",
            "list": "List available policy guides",
            "open": "Open a policy skill guide",
            "read": "Read a policy clause file",
            "check": "Check the delivery policy rules",
        }.get((arguments or {}).get("action"), "Consult the policy")
    return TOOL_TITLES.get(tool, tool.replace("_", " ").capitalize())


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
    strategy = ", ".join(f"{role_title(role)}: {strategy_title(name)}" for role, name in setup["strategy"].items())
    agents = "single agent" if setup["agents"] == "single" else "multi-agent"
    control = {"agentic": "Agentic control", "graph": "Graph-controlled workflow"}.get(setup["control"], setup["control"])
    policy = {"rag": "RAG policy", "skills": "Skills policy"}.get(setup["policy_mode"], setup["policy_mode"])
    return f"{setup['name']}-{setup['variant']} · {agents}, {control}, {policy} · {strategy}"


# ---------------------------------------------------------------- timeline


def show_tool_result(tool: str, output: dict[str, Any]) -> None:
    if "error" in output:
        st.error(output["error"])
    elif tool == "policy" and "content" in output:
        st.markdown(f"**Policy file:** `{output.get('skill', '')}/{output.get('file', '')}`")
        st.markdown(output["content"])
    elif tool == "policy" and "catalog" in output:
        st.dataframe(output["catalog"], hide_index=True, width="stretch")
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
        st.markdown(f"**Prompt guard (LLM) — screening result:** {event['label']}" +
                    (f" (prompt-injection score {score:.4f})" if score is not None else ""))
    elif kind == "agent_start":
        st.markdown(f"#### {role_title(event['agent'])} · {strategy_title(event['strategy'])}")
    elif kind == "node_start" and event.get("node") in NODE_TITLES:
        st.markdown(f"**{NODE_TITLES[event['node']]}**" + (f" · {role_title(event['agent'])}"
                                                          if event.get("agent") else ""))
    elif kind == "llm_response" and event.get("reasoning"):
        latency = event.get("latency_s")
        timing = f" · {latency:.2f}s" if isinstance(latency, (int, float)) else ""
        st.markdown(f"**{role_title(event['role'])} — reasoning{timing}**")
        st.info(event["reasoning"])
    elif kind == "terminal_tool_name_repaired":
        st.caption(f"Resolved `{event['generated']}` as `{event['resolved']}` using the terminal tool schema.")
    elif kind == "plan":
        label = "Plan remaining actions" if event.get("check") else "Plan actions before execution"
        actor = role_title(event.get("agent", "agent"))
        st.markdown(f"**{actor} — {label}:** " +
                    (" → ".join(tool_title(name, args) for name, args in event["calls"])
                     if event["calls"] else "No further actions planned"))
    elif kind == "plan_check":
        st.caption(f"{role_title(event.get('agent', 'agent'))} — check for unfinished work "
                   f"(completion check {event['check']})")
    elif kind == "plan_rejected":
        st.warning(f"{role_title(event.get('agent', 'agent'))} — revise the action plan "
                   f"(replan {event['replan']}): {event['reason']}")
    elif kind == "tool_call_requested":
        actor = role_title(event["agent"]) if event.get("agent") else "Workflow"
        if event["tool"] == "backend_write":
            actor = "Graph workflow"
        st.markdown(f"**{actor} — {tool_title(event['tool'], event['input'])}**")
        st.caption(f"Tool: {event['tool']} · Arguments: {json.dumps(event['input'])}")
    elif kind == "tool_end":
        output = event["output"]
        result = "Failed" if "error" in output else "Result received"
        if "error" not in output and output.get("status"):
            result = str(output["status"]).replace("_", " ").capitalize()
        st.caption(f"{tool_title(event['tool'])} — {result} · {event['elapsed_s']:.2f}s")
        show_tool_result(event["tool"], event["output"])
    elif kind == "control_call" and event["tool"] == "ask_specialist":
        st.markdown(f"**Supervisor Agent — delegate to {role_title(event['input'].get('specialist', 'specialist'))}:** "
                    f"{event['input'].get('request', '')}")
    elif kind == "control_call" and event["tool"] == "request_review":
        st.markdown(f"**{role_title(event.get('agent', 'agent'))} — request write approval from Reviewer (LLM)**")
        st.caption(f"Proposed write: {json.dumps(event['input'])}")
    elif kind == "control_call" and event["tool"] == "finish":
        st.markdown(f"**{role_title(event.get('agent', 'agent'))} — submit the task outcome:** "
                    f"{status_title(event['input'].get('outcome', ''))} · {event['input'].get('note')}")
    elif kind == "control_result" and event["tool"] == "ask_specialist" and "outcome" in event["output"]:
        st.caption(f"{role_title(event['output']['specialist'])} — report to Supervisor Agent: "
                   f"{status_title(event['output']['outcome'])} · "
                   f"{event['output'].get('note') or ''}")
    elif kind == "proposal":
        writes = "; ".join(f"{w['operation']} {w['order_id']}" + (f" {w['amount']}" if w.get("amount") else "")
                           for w in event.get("writes", []))
        st.markdown(f"**{role_title(event.get('agent', 'agent'))} — submit an action proposal:** "
                    f"{status_title(event['outcome'])} · {event['note']}" +
                    (f" · Proposed writes: {writes}" if writes else " · No business write proposed"))
    elif kind == "proposal_refused":
        st.warning(f"Graph workflow — return {role_title(event.get('agent', 'agent'))} "
                   f"to evidence gathering: {event['reason']}")
    elif kind == "approval":
        stage = {"write": "proposed write", "reply": "customer response"}.get(event['stage'], event['stage'])
        text = f"**Reviewer (LLM) — {stage} review:** {event['decision']} — {event['reason']}"
        if event["decision"] == "approve":
            st.success(text)
        else:
            st.warning(text + "".join(f"\n- {issue}" for issue in event["issues"]))
    elif kind == "write_result":
        result = event["result"]
        amount = f" {result['amount']} {result.get('currency', '')}" if result.get("amount") else ""
        st.success(f"Write executor — recorded {result['operation']}{amount} on "
                   f"{result['order_id']} ({result['write_id']})")
    elif kind == "reply_wording_rejected":
        st.warning("Response wording checks — request a revised draft: " + "; ".join(event["issues"]))
    elif kind == "repeated_error":
        st.error(f"`{event['tool']}` failed {event['times']} times in a row with the same error; the agent stopped")
    elif kind == "tool_retry":
        st.caption(f"`{event['tool']}` failed ({event['error'][:150]}), running it again")
    elif kind in {"json_retry", "json_failed"}:
        st.caption(f"{role_title(event['role'])} — invalid structured response "
                   f"(attempt {event['attempt']}): {event['error'][:200]}")
    elif kind == "provider_error":
        role = f" ({role_title(event['role'])})" if event.get("role") else ""
        st.error(f"LLM provider request{role} failed: {event.get('error', 'unknown provider error')}")
    elif kind == "turn_stopped":
        st.warning(f"Stopped for a person: {event['reason']}")
    elif kind == "budget_exceeded":
        st.error(f"Budget reached: {event['reason']}")
    elif kind == "handoff_created":
        st.warning(f"Handed off to a human: {event['packet'].get('reason')}")
    elif kind == "ticket_saved":
        st.caption(f"Ticket storage — saved the conversation · {status_title(event['status'])}")


def summary(response: dict[str, Any]) -> str:
    trace = response["trace"]
    strategy = ", ".join(f"{role_title(role)}: {strategy_title(name)}" for role, name in response["strategy"].items())
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
                st.markdown("**Specialist Agent reports**")
                st.dataframe([{"Agent": role_title(part.get("specialist", "specialist")),
                               "Task": part.get("request"), "Outcome": status_title(part.get("outcome", "")),
                               "Summary": part.get("note")}
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
        st.caption(f"Model: {info['models'].get('agent')} · provider {info['provider']}")

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
        with st.status("Processing the customer request…", expanded=True) as status:
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
            status.update(label="Customer response ready" if "response" in exchange else "Request failed",
                          state="complete" if "response" in exchange else "error")
    if "response" in exchange:
        st.session_state.ticket_id = exchange["response"]["ticket_id"]
        st.session_state.history = exchange["response"]["history"]
    exchanges.append(exchange)
    st.rerun()
