"""LangGraph multi-agent workflow (S3: supervisor, specialists, approver and gateway, with RAG policy).

guard → triage → specialist → decide → execute → reply
                     ↑            │
                     └─ approver reject (re-proposal)
Any node can divert to handoff, which writes the packet and then goes to reply (hold reply).

LLMs only classify, propose, review and write. The route tool, `TicketFlow` and `MockWrites`
decide what executes: no token without a route at least as strict as the rule engine's
minimum, and no write without a token bound to that ticket, order and amount.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from support_ops.agents.approver import Approver
from support_ops.agents.guard import classify
from support_ops.agents.models import ApprovalResult, HandoffPacket, Proposal, Triage
from support_ops.agents.runtime import AgentContext, Budget, BudgetExceeded, Retries, llm_from_config
from support_ops.agents.specialist import STRATEGIES, Specialist
from support_ops.agents.supervisor import Supervisor
from support_ops.agents.toolbox import ORDERS_SHIPPING_TOOLS, RETURNS_REFUNDS_TOOLS, Toolbox, policy_search_tool
from support_ops.backend.postgres import PostgresReader
from support_ops.gateway.approvals import TicketFlow
from support_ops.gateway.route import RouteTool
from support_ops.gateway.ticket_state import TicketState
from support_ops.gateway.writes import MockWrites
from support_ops.helpers.cancellation import CancellationToken
from support_ops.helpers.clock import Clock
from support_ops.helpers.llm import ChatResult, LLM
from support_ops.helpers.tracing import Tracer
from support_ops.policy.provider import PolicyProvider
from support_ops.policy.routing import RoutingRules
from support_ops.policy.rules import Rules
from support_ops.policy.rag import RagPolicy
from support_ops.systems.tools import SupportTools

SPECIALIST_TOOLS = {"orders_shipping": ORDERS_SHIPPING_TOOLS, "returns_refunds": RETURNS_REFUNDS_TOOLS}
DEFAULTS = {
    "triage_confidence_min": 0.5,   # below this, hand off rather than guess the specialist
    "max_reproposals": 1,           # approver rejections the specialist may fix
    "max_reply_drafts": 2,          # drafts the approver may review before falling back
    "policy_sections": {"orders_shipping": ["Orders and shipping"], "returns_refunds": ["Returns and refunds"]},
}
# Earlier turns arrive with each request (the API is stateless).
HISTORY_ROLES = {"customer", "agent"}
HOLD_REPLY = "I’ve referred your request to a support specialist for review."
# Handoffs from these triggers get the fixed hold reply with no further LLM calls.
FIXED_REPLY_TRIGGERS = ("guard", "budget")


class MultiAgentState(TypedDict, total=False):
    ticket_id: str
    customer_message: str
    history: list[dict[str, str]]
    strategy: str
    next: str
    triage: Triage | None
    proposal: Proposal | None
    min_route: dict[str, Any] | None
    approval: ApprovalResult | None
    feedback: list[str]
    reproposals: int
    evidence: list[dict[str, Any]]
    reflections: list[str]
    execution: dict[str, Any] | None
    customer_question: str | None
    handoff_reason: str | None
    handoff_packet: dict[str, Any] | None
    reply: str
    reply_review: ApprovalResult | None


class MultiAgentWorkflow:
    def __init__(self, llm: LLM, cfg: dict[str, Any], *, reader: PostgresReader | None = None,
                 policy: PolicyProvider | None = None, clock: Clock | None = None, tracer: Tracer | None = None,
                 cancellation: CancellationToken | None = None):
        if cfg["policy"].get("mode") != "rag":
            raise ValueError("the multi-agent arm is S3 only: set policy.mode to rag")
        self.llm, self.cfg = llm, cfg
        self.settings = {**DEFAULTS, **cfg.get("multi_agent", {})}
        self.retries = Retries.from_config(cfg)
        self.clock = clock or Clock(datetime.fromisoformat(cfg["run"]["clock_start"]))
        self._provided_tracer = tracer
        # Checked at every logged step (each LLM call, tool call and node), as in the single agent.
        self.cancellation = cancellation or CancellationToken()
        rules = Rules.load(cfg["policy"]["rules_file"])
        self.routing = RoutingRules.load(cfg["policy"]["rules_file"])
        self.reader = reader or PostgresReader()
        self.ticket_state = TicketState(self.clock, rules=rules)
        self.tools = SupportTools(self.reader, policy or RagPolicy.from_config(cfg, rules), self.ticket_state)
        self.flow = TicketFlow(self.ticket_state, self.clock)
        self.writes = MockWrites(self.flow, self.routing.currency)
        self._tokens: dict[str, str | None] = {}   # kept out of graph state, so never returned
        graph = StateGraph(MultiAgentState)
        nodes = ("guard", "triage", "specialist", "decide", "execute", "handoff", "reply")
        for name in nodes:
            graph.add_node(name, self._guarded(name, getattr(self, f"_{name}")))
            if name != "reply":
                graph.add_conditional_edges(name, lambda state: state["next"], list(nodes))
        graph.add_edge(START, "guard")
        graph.add_edge("reply", END)
        self.graph = graph.compile()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **kwargs: Any) -> MultiAgentWorkflow:
        return cls(llm_from_config(cfg), cfg, **kwargs)

    # ------------------------------------------------------------- run

    def invoke(self, ticket_id: str, customer_message: str, strategy: str | None = None,
               tracer: Tracer | None = None, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        selected = strategy or self.cfg.get("agent", {}).get("strategy", "react")
        if selected not in STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}")
        history = [{"role": turn["role"], "content": turn["content"]} for turn in history or []]
        if any(turn["role"] not in HISTORY_ROLES for turn in history):
            raise ValueError("history roles must be customer or agent")
        self.tracer = tracer or self._provided_tracer or Tracer(ticket_id)
        self.ctx = AgentContext(self._chat, self._event, self.clock, Budget(self.cfg["limits"], self.tracer),
                                self.retries)
        self.supervisor, self.approver = Supervisor(self.ctx), Approver(self.ctx)
        self._event("workflow_start", system="multi_agent", strategy=selected, history_turns=len(history))
        state = self.graph.invoke({
            "ticket_id": ticket_id, "customer_message": customer_message, "history": history,
            "strategy": selected, "triage": None, "proposal": None, "min_route": None, "approval": None,
            "feedback": [], "reproposals": 0, "evidence": [], "reflections": [], "execution": None,
            "customer_question": None, "handoff_reason": None, "handoff_packet": None,
        }, {"recursion_limit": 40})
        status = self.flow.status(ticket_id)
        self._event("workflow_end", status=status, llm_calls=self.tracer.llm_calls,
                    elapsed_s=round(self.tracer.elapsed_s, 4))
        dump = lambda value: value.model_dump(mode="json") if value is not None else None
        return {
            "ticket_id": ticket_id, "strategy": selected, "reply": state.get("reply", ""),
            "ticket_status": status, "handoff_packet": state.get("handoff_packet"),
            "triage": dump(state.get("triage")), "proposal": dump(state.get("proposal")),
            "min_route": state.get("min_route"), "approval": dump(state.get("approval")),
            "reply_review": dump(state.get("reply_review")), "execution": state.get("execution"),
            "evidence": state.get("evidence", []), "reflections": state.get("reflections", []),
            "audit_log": list(self.flow.audit_log), "history": history, "trace": self.tracer,
        }

    def _event(self, kind: str, **payload: Any) -> None:
        self.cancellation.check()
        self.tracer.log(kind, simulated_at=self.clock.now().isoformat(), **payload)

    def _chat(self, role: str, messages: list[dict[str, Any]], *, tools=None, json_mode=False) -> ChatResult:
        self._event("llm_request", role=role)
        result = self.llm.chat(role, messages, tools=tools, tracer=self.tracer, json_mode=json_mode)
        self._event("llm_response", role=role, tool_calls=[call.name for call in result.tool_calls],
                    **({"reasoning": result.reasoning} if result.reasoning else {}))
        return result

    def _guarded(self, name: str, node: Callable[[MultiAgentState], dict[str, Any]]):
        """A budget hit in any node hands the ticket off instead of failing the request."""
        def run(state: MultiAgentState) -> dict[str, Any]:
            try:
                return node(state)
            except BudgetExceeded as exc:
                if name in {"handoff", "reply"}:
                    raise
                return {"next": "handoff", "handoff_reason": f"budget: {exc}"}
        return run

    def _brief(self, state: MultiAgentState) -> dict[str, Any]:
        brief: dict[str, Any] = {"message": state["customer_message"]}
        if state.get("history"):
            brief["history"] = state["history"]
        if state.get("triage"):
            brief["triage"] = state["triage"].model_dump(mode="json")
        if state.get("feedback"):
            brief["approver_feedback"] = state["feedback"]
        return brief

    @staticmethod
    def _to_handoff(reason: str) -> dict[str, Any]:
        return {"next": "handoff", "handoff_reason": reason}

    # ------------------------------------------------------------- nodes

    def _guard(self, state: MultiAgentState) -> dict[str, Any]:
        self._event("node_start", node="guard", ticket_id=state["ticket_id"])
        flag = classify(self._chat, self.cfg, state["customer_message"], self._event)
        return {"next": "triage"} if flag == "benign" else self._to_handoff(f"guard: {flag}")

    def _triage(self, state: MultiAgentState) -> dict[str, Any]:
        try:
            triage = self.supervisor.triage(self._brief(state))
        except ValueError as exc:
            return self._to_handoff(f"triage failed: {exc}")
        self.flow.transition(state["ticket_id"], "triaged", intent=triage.intent, specialist=triage.specialist)
        update: dict[str, Any] = {"triage": triage}
        if triage.wants_human:
            return {**update, **self._to_handoff("customer asked for a human")}
        if triage.duplicate_of:
            return {**update, **self._to_handoff(f"duplicate of open ticket {triage.duplicate_of}")}
        if triage.specialist is None:
            return {**update, **self._to_handoff("no specialist covers this request")}
        if triage.confidence < self.settings["triage_confidence_min"]:
            return {**update, **self._to_handoff(f"low triage confidence ({triage.confidence:.2f})")}
        return {**update, "next": "specialist"}

    def _specialist(self, state: MultiAgentState) -> dict[str, Any]:
        name = state["triage"].specialist
        route = RouteTool(self.reader, self.routing, self.clock, name)
        reads = [tool for tool in SPECIALIST_TOOLS[name] if tool != "policy"]
        sections = self.settings["policy_sections"].get(name)
        self._check_sections(name, sections)
        toolbox = Toolbox(self.tools, reads, {"policy": policy_search_tool(self.tools, sections),
                                              "route": route.as_tool()})
        result = Specialist(name, toolbox, self.ctx, self.cfg).run(
            state["ticket_id"], self._brief(state), state["strategy"])
        update: dict[str, Any] = {
            "evidence": [*state["evidence"], *result.evidence],
            "reflections": [*state["reflections"], *result.reflections],
            "proposal": result.proposal, "min_route": result.min_route,
        }
        if result.error:
            reason = f"budget: {result.error}" if "budget" in result.error or "limit" in result.error \
                else f"specialist: {result.error}"
            return {**update, **self._to_handoff(reason)}
        proposal = result.proposal
        if proposal.action == "none" and proposal.customer_question:
            self.flow.transition(state["ticket_id"], "awaiting_customer")
            return {**update, "customer_question": proposal.customer_question, "next": "reply"}
        self.flow.transition(state["ticket_id"], "proposed", route=proposal.route, operation=proposal.action)
        return {**update, "next": "decide"}

    def _check_sections(self, name: str, sections: list[str] | None) -> None:
        """Flag configured sections the corpus does not have, so an empty search is never a silent mystery."""
        retriever = getattr(self.tools.policy, "retriever", None)
        try:
            known = {section.casefold() for section in retriever.sections()} if retriever else None
        except Exception:
            known = None
        if known and (missing := [section for section in sections or [] if section.casefold() not in known]):
            self._event("policy_sections_missing", specialist=name, missing=missing, corpus_sections=sorted(known))

    def _decide(self, state: MultiAgentState) -> dict[str, Any]:
        self._event("node_start", node="decide", route=state["proposal"].route)
        ticket_id, proposal, minimum = state["ticket_id"], state["proposal"], state["min_route"]
        if proposal.route == "human":
            reasons = "; ".join(minimum.get("reasons", [])) if minimum else ""
            return self._to_handoff("route: human" + (f" ({reasons})" if reasons else ""))
        approval = None
        if proposal.route == "approver":
            approval = self.approver.check_proposal(self._brief(state), proposal, minimum, state["evidence"])
            if approval.decision == "reject" and state["reproposals"] < self.settings["max_reproposals"]:
                self.flow.transition(ticket_id, "triaged", reason="approver rejected the proposal")
                return {"approval": approval, "feedback": approval.issues,
                        "reproposals": state["reproposals"] + 1, "next": "specialist"}
            if approval.decision != "approve":
                return {"approval": approval,
                        **self._to_handoff(f"approver {approval.decision}: {'; '.join(approval.issues)}")}
        try:
            granted = self.flow.approve(ticket_id, proposal, minimum, approval)
        except ValueError as exc:
            return {"approval": approval, **self._to_handoff(f"gateway refused: {exc}")}
        self._tokens[ticket_id] = granted["approval_token"]
        return {"approval": approval, "next": "execute" if granted["approval_token"] else "reply"}

    def _execute(self, state: MultiAgentState) -> dict[str, Any]:
        self._event("node_start", node="execute")
        ticket_id, proposal = state["ticket_id"], state["proposal"]
        result = self.writes.execute(
            ticket_id, token=self._tokens.pop(ticket_id, None) or "", operation=proposal.action,
            order_id=proposal.order_id, amount=proposal.amount,
            idempotency_key=f"{ticket_id}:{proposal.action}:{proposal.order_id}:{proposal.amount}")
        self._event("write_result", result=result)
        if result["status"] != "executed":
            return {"execution": result, **self._to_handoff(f"write refused: {result.get('reason')}")}
        return {"execution": result, "next": "reply"}

    def _handoff(self, state: MultiAgentState) -> dict[str, Any]:
        self._event("node_start", node="handoff", reason=state["handoff_reason"])
        proposal, triage = state.get("proposal"), state.get("triage")
        actions = [f"{item['agent']}: {item['tool']}" + (" (error)" if "error" in item["result"] else "")
                   for item in state["evidence"]]
        if state.get("execution"):
            actions.append(f"write {state['execution']['status']}: {state['execution'].get('operation')}")
        packet = HandoffPacket(
            ticket_id=state["ticket_id"], reason=state["handoff_reason"],
            summary=f"{triage.intent if triage else 'request'}: {state['customer_message'][:500]}",
            actions_taken=actions, cited_clauses=proposal.cited_clauses if proposal else [],
            recommendation=(f"Specialist proposes {proposal.action} via {proposal.route}: {proposal.reason}"
                            if proposal else "Review the request; the automated flow stopped before a proposal."),
            proposal=proposal, min_route=state.get("min_route"),
        )
        if self.flow.status(state["ticket_id"]) != "awaiting_human":
            self.flow.handoff(packet)
        self._event("handoff_created", packet=packet.model_dump(mode="json"))
        return {"handoff_packet": packet.model_dump(mode="json"), "next": "reply"}

    @staticmethod
    def _verified_facts(state: MultiAgentState) -> list[dict[str, Any]]:
        """Successful read and policy results, so the reply rests on what the tools returned."""
        return [{"tool": item["tool"], "result": item["result"]} for item in state["evidence"]
                if item["tool"] != "route" and "error" not in item["result"]]

    def _customer_outcome(self, state: MultiAgentState, status: str) -> dict[str, Any]:
        """What the reply may say, in customer terms. Routes, approvals, rule names, thresholds and
        clause or item IDs stay internal, so the reply cannot leak or misread them."""
        facts = self._verified_facts(state)
        names = {item["id"]: item.get("name") for fact in facts if fact["tool"] == "get_items_for_order"
                 for item in fact["result"].get("items", [])}
        proposal, execution = state.get("proposal"), state.get("execution")
        executed = bool(execution and execution.get("status") == "executed")
        outcome: dict[str, Any] = {"status": {"approved": "resolved", "executed": "resolved"}.get(status, status),
                                   "facts": facts}
        if proposal and proposal.action != "none":
            request: dict[str, Any] = {"action": proposal.action, "order_id": proposal.order_id}
            if proposal.amount is not None:
                request["amount"] = f"{proposal.amount:.2f} {self.routing.currency}"
            if proposal.items:
                request["items"] = [{"product": names.get(line.item_id) or "the item", "quantity": line.quantity}
                                    for line in proposal.items]
            outcome["request"] = request
        if proposal:
            # What the specialist found, e.g. that a shipment is past its estimated date.
            outcome["specialist_finding"] = proposal.reason
        if (state.get("min_route") or {}).get("overdue_shipment"):
            # Found by code from the shipment dates, so the apology does not depend on the model's arithmetic.
            outcome["problems"] = ["The shipment is past its estimated delivery date."]
        if outcome["status"] == "awaiting_human" and facts:
            outcome["tell_customer"] = (
                "Share what the facts show that answers the customer (e.g. the current status and "
                "estimated dates). Acknowledge every item in problems, if any, and apologise; likewise "
                "anything else that has gone wrong, such as a damaged item. Then say a support specialist will review the request and "
                "follow up. Promise no outcome, refund, compensation or timeline.")
        elif outcome["status"] == "awaiting_human":
            outcome["tell_customer"] = "A support specialist will review the request. Promise nothing else."
        elif outcome["status"] == "awaiting_customer":
            outcome["tell_customer"] = f"Ask: {state.get('customer_question')}"
        elif executed:
            outcome["tell_customer"] = (f"The {proposal.action} is recorded and complete; nothing is pending "
                                        "review. Say what was recorded.")
        else:
            outcome["tell_customer"] = ("Answer the question from the facts. Acknowledge every item in "
                                        "problems, if any, and apologise. No action was taken.")
        return outcome

    def _reply(self, state: MultiAgentState) -> dict[str, Any]:
        ticket_id = state["ticket_id"]
        reason = state.get("handoff_reason") or ""
        if reason.startswith(FIXED_REPLY_TRIGGERS):
            return {"reply": HOLD_REPLY}
        status = self.flow.status(ticket_id)
        outcome = self._customer_outcome(state, status)
        brief, issues, review = self._brief(state), None, None
        try:
            for _ in range(self.settings["max_reply_drafts"]):
                reply = self.supervisor.draft_reply(brief, outcome, issues)
                review = self.approver.check_reply(brief, reply, outcome)
                if review.decision == "approve" and reply:
                    if status in {"approved", "executed"}:
                        self.flow.transition(ticket_id, "resolved")
                    return {"reply": reply, "reply_review": review}
                issues = review.issues or ["empty reply"]
        except BudgetExceeded:
            pass
        # No reply passed review: send the fixed hold reply and let an operator follow up.
        if self.flow.status(ticket_id) != "awaiting_human":
            update = self._handoff({**state, "handoff_reason": "reply failed review"})
            return {"reply": HOLD_REPLY, "reply_review": review, "handoff_packet": update["handoff_packet"],
                    "handoff_reason": "reply failed review"}
        return {"reply": HOLD_REPLY, "reply_review": review}
