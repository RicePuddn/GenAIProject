"""Demo page for the multi-agent arms (S3 rag, S4 skills): chat as the customer and watch each agent work.

A client of the stateless multi-agent API (`support_ops.api_multi`): it keeps the conversation and
sends earlier turns as `history`, and reads /chat/stream so triage, specialist tool calls,
approvals, the write and any handoff appear while the ticket runs. Start the API
first, then from the repo root:

    .venv/bin/uvicorn support_ops.api_multi:app --host 127.0.0.1 --port 8001
    .venv/bin/streamlit run scripts/demo_multi_agent.py

Set SUPPORT_OPS_MULTI_API to use an API somewhere other than http://127.0.0.1:8001.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Iterator

import streamlit as st

API = os.environ.get("SUPPORT_OPS_MULTI_API", "http://127.0.0.1:8001").rstrip("/")
STRATEGIES = ["react", "plan_and_execute"]
DESIGNS = {"workflow": "Fixed workflow", "supervisor": "Pure supervisor"}
STATUS_LABELS = {
    "resolved": ("success", "Resolved"),
    "awaiting_customer": ("info", "Waiting for the customer"),
    "awaiting_human": ("warning", "Awaiting a human operator"),
}
NODE_TITLES = {
    "triage": "Supervisor · triage",
    "decide": "Gateway · route decision",
    "execute": "Gateway · mock write",
    "handoff": "Handoff to a human",
    "draft_reply": "Supervisor · drafting the reply",
}


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
            f"cannot reach the multi-agent API at {API} ({exc.reason}). Is `uvicorn support_ops.api_multi:app "
            "--port 8001` running? It accepts requests once the RAG models have loaded; then refresh."
        ) from exc


def call_api(path: str, timeout: float = 10) -> dict[str, Any]:
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


# ---------------------------------------------------------------- timeline


def show_tool_result(tool: str, output: dict[str, Any]) -> None:
    if "error" in output:
        st.error(output["error"])
    elif tool == "policy" and "clauses" in output:
        scope = (f" · scope: own sections ({', '.join(output['sections'])})" if output.get("scope") == "own"
                 else " · scope: all sections")
        hidden = f" · {output['filtered_out']} from other sections hidden" if output.get("filtered_out") else ""
        st.caption(f"Retrieved for query: “{output.get('query', '')}”{scope}{hidden}")
        if output.get("note"):
            st.info(output["note"])
        st.dataframe(
            [{"clause": c["id"], "title": c["title"], "score": c["score"], "via": c["via"], "text": c["text"]}
             for c in output["clauses"]],
            hide_index=True, width="stretch",
        )
    elif tool == "route" and output.get("status") == "recorded":
        minimum = output["min_route"]
        st.success(f"Proposal recorded on route **{output['route']}** (rules minimum: {minimum['route']})")
        if minimum.get("reasons"):
            st.caption("Rule engine: " + "; ".join(minimum["reasons"]))
    else:
        st.json(output, expanded=False)


def show_event(event: dict[str, Any]) -> None:
    """One trace event as a timeline step; used both live and when redrawing earlier turns."""
    kind = event["kind"]
    if kind == "guard_classification":
        score = event.get("malicious_score")
        st.markdown(f"**Guard:** {event['label']}" + (f" (malicious score {score:.4f})" if score is not None else ""))
    elif kind == "node_start" and event.get("node") == "specialist":
        st.markdown(f"#### Specialist · {event['specialist'].replace('_', ' ')} ({event['strategy']})")
    elif kind == "node_start" and event.get("node") == "approver":
        stage = "proposal check" if event.get("stage") == "pre_execution" else "reply check"
        st.markdown(f"#### Approver · {stage}")
    elif kind == "node_start" and event.get("node") in NODE_TITLES:
        st.markdown(f"#### {NODE_TITLES[event['node']]}")
    elif kind == "triage":
        specialists = " + ".join(task["specialist"].replace("_", " ") for task in event.get("tasks") or []) \
            or (event["specialist"] or "none")
        st.markdown(f"**Intent:** {event['intent']} · **specialist:** {specialists} · "
                    f"**urgency:** {event['urgency']} · **confidence:** {event['confidence']:.2f}"
                    + (" · **wants a human**" if event.get("wants_human") else ""))
    elif kind == "llm_response" and event.get("reasoning"):
        with st.expander(f"Thought ({event['role']})"):
            st.write(event["reasoning"])
    elif kind == "plan":
        st.markdown("**Plan:** " + " → ".join(f"`{name}`" for name, _ in event["calls"]) if event["calls"]
                    else "**Plan:** no calls needed")
    elif kind == "plan_route_dropped":
        st.caption("The plan included the route tool; dropped (the proposal comes after the plan runs)")
    elif kind == "plan_rejected":
        st.warning(f"Plan rejected: {event['reason']}")
    elif kind == "tool_call_requested":
        st.markdown(f"**Tool:** `{event['tool']}` `{json.dumps(event['input'])}`")
    elif kind == "tool_end":
        st.caption(f"`{event['tool']}` finished in {event['elapsed_s']:.2f}s")
        show_tool_result(event["tool"], event["output"])
    elif kind == "approval":
        text = f"**Verdict:** {event['decision']} — {event['reason']}"
        if event["decision"] == "approve":
            st.success(text)
        else:
            st.warning(text + "".join(f"\n- {issue}" for issue in event["issues"]))
    elif kind == "write_result":
        result = event["result"]
        if result["status"] == "executed":
            amount = f" {result['amount']} {result['currency']}" if result.get("amount") else ""
            st.success(f"Mock {result['operation']}{amount} on {result['order_id']} executed ({result['write_id']})")
        else:
            st.error(f"Write refused: {result.get('reason')}")
    elif kind == "order_id_filled":
        st.caption(f"The proposal named no order, so the one looked up ({event['order_id']}) was checked")
    elif kind == "orchestrator_step":
        if event["action"] == "delegate":
            st.markdown(f"#### Supervisor · next step: {event['specialist'].replace('_', ' ')}")
            st.caption(f"{event['request']} ({event['reason']})")
        else:
            st.markdown("#### Supervisor · finish")
            st.caption(event["reason"])
    elif kind == "supervisor_step":
        arguments = ", ".join(f"{key}: {value}" for key, value in event["arguments"].items())
        st.markdown(f"#### Supervisor · {event['tool'].replace('_', ' ')}")
        if arguments:
            st.caption(arguments)
    elif kind == "supervisor_plan":
        st.markdown("#### Supervisor · " + ("plan again" if event.get("again") else "plan"))
        st.caption(" → ".join(step["tool"].replace("_", " ") for step in event["steps"]))
    elif kind == "supervisor_replan":
        st.warning(f"Planning again ({event['replans']}): {event['reason']}")
    elif kind == "supervisor_refused":
        st.error(f"Refused {'`' + event['tool'] + '`' if event.get('tool') else 'the step'}: {event['reason']}")
    elif kind == "supervisor_limit":
        st.warning(f"The supervisor stopped after {event['steps']} steps")
    elif kind == "orchestrator_refused":
        st.warning(f"Delegation refused: {event['reason']}")
    elif kind == "orchestration_limit":
        st.warning(f"Orchestration stopped after {event['steps']} steps")
    elif kind == "part_start":
        st.markdown(f"### Part {event['part']} of {event['of']}: {event['request'] or 'the whole message'}")
    elif kind == "part_done":
        st.caption(f"Part {event['part']} ({event['specialist'].replace('_', ' ')}) ended: {event['status']}; "
                   "on to the next part")
    elif kind == "off_topic":
        st.info("Not a support request: replying with what we can help with (no specialist, no handoff)")
    elif kind == "missing_record":
        st.info(f"Not found: {', '.join(event['records'])}. Asking the customer to check: {event['question']}")
    elif kind == "repeated_error":
        st.error(f"`{event['tool']}` failed {event['times']} times in a row with the same error; attempt stopped")
    elif kind == "tool_retry":
        st.caption(f"`{event['tool']}` failed ({event['error'][:150]}), running it again (attempt {event['attempt'] + 1})")
    elif kind == "json_retry":
        st.caption(f"Malformed reply from the {event['role']}, retrying (attempt {event['attempt']}): {event['error'][:200]}")
    elif kind == "json_failed":
        st.error(f"The {event['role']} gave no valid reply: {event['error'][:300]}")
    elif kind == "specialist_end" and event.get("error"):
        st.error(f"Specialist stopped: {event['error']}")
    elif kind == "handoff_created":
        st.warning(f"Handed off: {event['packet'].get('reason')}")


# ---------------------------------------------------------------- outcome


def proposal_row(proposal: dict[str, Any], minimum: dict[str, Any] | None, status: str | None = None) -> dict[str, Any]:
    return {
        "specialist": proposal["specialist"], **({"part ended": status} if status else {}),
        "action": proposal["action"], "order": proposal.get("order_id"), "amount": proposal.get("amount"),
        "items": ", ".join(f"{line['item_id']} ×{line['quantity']}" for line in proposal.get("items", [])),
        "route": proposal["route"], "rules minimum": minimum["route"] if minimum else None,
        "cited": ", ".join(proposal.get("cited_clauses", [])), "reason": proposal["reason"],
    }


def show_outcome(response: dict[str, Any]) -> None:
    parts = response.get("parts") or []
    if len(parts) > 1:
        st.markdown("**Proposals, one per part**")
        st.dataframe([proposal_row(part["proposal"], part.get("min_route"), part["status"]) if part.get("proposal")
                      else {"specialist": part["specialist"], "part ended": part["status"],
                            "reason": part.get("handoff_reason") or part.get("customer_question")}
                      for part in parts], hide_index=True, width="stretch")
    elif proposal := response.get("proposal"):
        st.markdown("**Proposal**")
        st.dataframe([proposal_row(proposal, response.get("min_route"))], hide_index=True, width="stretch")
    if response.get("audit_log"):
        st.markdown("**Audit log**")
        st.dataframe([{
            "at": event.get("at", "")[11:19], "event": event.get("type") or event.get("event"),
            "outcome": event.get("outcome", ""),
            "detail": " → ".join(filter(None, [event.get("from"), event.get("to")])) or event.get("reason", ""),
        } for event in response["audit_log"]], hide_index=True, width="stretch")
    if response.get("handoff_packet"):
        st.markdown("**Handoff packet**")
        st.json(response["handoff_packet"], expanded=False)


def summary(response: dict[str, Any]) -> str:
    trace = response["trace"]
    design = DESIGNS.get(response.get("design", "workflow"), "Fixed workflow").lower()
    orchestration = response.get("orchestration", response["strategy"])
    return (f"What the agents did · {design} · supervisor {orchestration} · "
            f"specialists {response['strategy']} · {trace['tool_call_count']} tools · "
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
            timeline, outcome = st.tabs(["Timeline", "Outcome"])
            with timeline:
                for event in trace["events"]:
                    show_event(event)
            with outcome:
                show_outcome(response)


# ---------------------------------------------------------------- page

st.set_page_config(page_title="E-Commerce Support · Multi-agent", layout="wide")
if "history" not in st.session_state:
    new_ticket()

with st.sidebar:
    st.header("E-Commerce Support")
    try:
        info = call_api("/info", timeout=10)
        arm = "S4, with skills policy" if info["policy_mode"] == "skills" else "S3, with RAG policy"
        st.caption(f"Multi-agent ({arm}): supervisor, specialists, approver and gateway")
        models = info["models"]
        st.markdown(f"**Policy:** {info['policy_mode']} ({info['corpus']})  \n"
                    f"**Supervisor:** {models['supervisor']}  \n**Specialist:** {models['specialist']}  \n"
                    f"**Approver:** {models['approver']} · effort {info['reasoning_effort']}")
    except RuntimeError as exc:
        info = {"default_strategy": "react"}
        st.error(str(exc))
    default = info.get("default_strategy", "react")
    default_design = info.get("default_design", "workflow")
    design = st.radio("Multi-agent design", list(DESIGNS), format_func=DESIGNS.get,
                      index=list(DESIGNS).index(default_design) if default_design in DESIGNS else 0,
                      help="Fixed workflow: code runs triage → specialist → decide → execute. Pure supervisor: "
                           "the supervisor chooses every step; the gateway refuses unsafe ones.")
    default_orchestration = info.get("default_orchestration", default)
    orchestration = st.radio("Supervisor strategy", STRATEGIES,
                             index=STRATEGIES.index(default_orchestration) if default_orchestration in STRATEGIES else 0,
                             help="react: decides each step after seeing the last result. plan_and_execute: "
                                  "plans the steps first (fixed workflow: the parts at triage) and plans again "
                                  "when a step goes differently.")
    strategy = st.radio("Specialists", STRATEGIES, index=STRATEGIES.index(default) if default in STRATEGIES else 0,
                        help="react: one tool call per turn. plan_and_execute: plan the calls, then run them.")
    st.button("New ticket", on_click=new_ticket, width="stretch")
    st.caption(f"Ticket: {st.session_state.ticket_id or 'new'} · {len(st.session_state.history)} turns sent as history")
    st.caption("Return windows are counted from today: try orders delivered in the last 30 days, "
               "e.g. ORD-2026-00031, ORD-2026-00040, ORD-2026-00041.")

exchanges = st.session_state.exchanges
for index, exchange in enumerate(exchanges):
    show_exchange(exchange, expanded=index == len(exchanges) - 1)

if message := st.chat_input("Write as the customer, e.g. One bottle in ORD-2026-00031 arrived dented."):
    exchange: dict[str, Any] = {"message": message}
    payload = {"customer_message": message, "strategy": strategy, "orchestration": orchestration, "design": design,
               "history": st.session_state.history}
    if st.session_state.ticket_id:
        payload["ticket_id"] = st.session_state.ticket_id
    with st.chat_message("user"):
        st.markdown(message)
    with st.chat_message("assistant"):
        # Steps are drawn as their events arrive; the page is redrawn in full once the turn ends.
        with st.status("Agents are working…", expanded=True) as status:
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
