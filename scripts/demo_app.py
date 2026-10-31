"""Demo page: chat with the support agent and watch each step live.

It is an ordinary client of the stateless API: it keeps the conversation and sends the earlier
turns as `history` with every message, and reads /chat/stream so the agent's thoughts and tool
calls appear while it works. Start the API first, then from the repo root:

    .venv/Scripts/streamlit run scripts/demo_app.py      (Windows; .venv/bin/... elsewhere)

Set SUPPORT_OPS_API to use an API somewhere other than http://127.0.0.1:8000.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Iterator

import streamlit as st

API = os.environ.get("SUPPORT_OPS_API", "http://127.0.0.1:8000").rstrip("/")


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


def call_api(path: str, timeout: float = 10) -> dict[str, Any]:
    with open_api(path, None, timeout) as response:
        return json.loads(response.read())


def stream_chat(payload: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Read Server-Sent Events from /chat/stream as (event, data) pairs, as they arrive."""
    with open_api("/chat/stream", payload, timeout=300) as response:
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


def show_tool_result(tool: str, output: dict[str, Any]) -> None:
    if tool == "policy" and "clauses" in output:
        st.caption(f"Retrieved for query: “{output.get('query', '')}”")
        st.dataframe(
            [{"clause": c["id"], "title": c["title"], "score": c["score"], "via": c["via"], "text": c["text"]}
             for c in output["clauses"]],
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
    elif kind == "llm_response" and event.get("role") == "supervisor" and event.get("reasoning"):
        st.markdown("**Thought**")
        st.info(event["reasoning"])
    elif kind == "tool_call_requested":
        st.markdown(f"**Tool:** `{event['tool']}` `{json.dumps(event['input'])}`")
    elif kind == "tool_end":
        st.caption(f"`{event['tool']}` finished in {event['elapsed_s']:.2f}s")
        show_tool_result(event["tool"], event["output"])
    elif kind == "plan_rejected":
        st.warning(f"Plan rejected: {event['reason']}")
    elif kind == "handoff_created":
        st.warning(f"Handed off to a human: {event['packet'].get('reason')}")


def summary(response: dict[str, Any]) -> str:
    trace = response["trace"]
    return (f"What the agent did · {response['strategy']} · {trace['tool_call_count']} tools · "
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
        trace = response["trace"]
        if response["ticket_status"] == "awaiting_human":
            st.warning("Ticket is awaiting a human operator.")
        with st.expander(summary(response), expanded=expanded):
            columns = st.columns(4)
            columns[0].metric("LLM calls", trace["llm_calls"])
            columns[1].metric("Tokens in / out", f"{trace['input_tokens']} / {trace['output_tokens']}")
            columns[2].metric("Cost (USD)", f"{trace['cost_usd']:.4f}")
            columns[3].metric("Time (s)", f"{trace['elapsed_s']:.1f}")
            for event in trace["events"]:
                show_event(event)
            if response.get("handoff_packet"):
                st.markdown("**Handoff packet**")
                st.json(response["handoff_packet"], expanded=False)


st.set_page_config(page_title="E-Commerce Support", layout="wide")
if "history" not in st.session_state:
    new_ticket()

with st.sidebar:
    st.header("E-Commerce Support")
    try:
        info = call_api("/info", timeout=10)
        st.markdown(f"**Policy:** {info['policy_mode']} ({info['corpus']})  \n"
                    f"**Model:** {info['model']} · effort {info['reasoning_effort']}")
    except RuntimeError as exc:
        info = {"default_strategy": "react"}
        st.error(str(exc))
    try:
        accounts = call_api("/customers", timeout=10)
    except RuntimeError as exc:
        accounts = []
        st.warning(f"Accounts unavailable, chatting signed out: {exc}")
    names = {row["id"]: f"{row['first_name']} {row['last_name']} ({row['id']})" for row in accounts}
    # Switching account starts a new ticket, so one conversation never spans two customers.
    customer_id = st.selectbox("Signed in as", [None, *names], key="customer_id", on_change=new_ticket,
                               format_func=lambda account: names.get(account, "Not signed in"),
                               help="The agent sees and changes only this customer's orders, and the "
                                    "ticket is saved as theirs. Signed out, nothing is scoped.")
    strategies = ["react", "plan_and_execute"]
    strategy = st.radio("Strategy", strategies, index=strategies.index(info.get("default_strategy", "react")))
    st.button("New ticket", on_click=new_ticket, width="stretch")
    st.caption(f"Ticket: {st.session_state.ticket_id or 'new'} · {len(st.session_state.history)} turns sent as history")
    st.caption("Policy mode is server config (`policy.mode` in configs/base.yaml); restart uvicorn to switch.")

exchanges = st.session_state.exchanges
for index, exchange in enumerate(exchanges):
    show_exchange(exchange, expanded=index == len(exchanges) - 1)

if message := st.chat_input("Write as the customer, e.g. Can I return order ORD-2026-00041?"):
    exchange: dict[str, Any] = {"message": message}
    payload = {"customer_message": message, "strategy": strategy, "history": st.session_state.history}
    if st.session_state.ticket_id:
        payload["ticket_id"] = st.session_state.ticket_id
    if customer_id:
        payload["customer_id"] = customer_id
    with st.chat_message("user"):
        st.markdown(message)
    with st.chat_message("assistant"):
        # Steps are drawn as their events arrive; the page is redrawn in full once the turn ends.
        with st.status("Agent is working…", expanded=True) as status:
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
