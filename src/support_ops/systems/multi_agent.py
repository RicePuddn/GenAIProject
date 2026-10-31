"""LangGraph multi-agent workflow (supervisor, specialists, approver and gateway; S3 with RAG policy, S4 with skills).

Two independent settings, each react or plan_and_execute:

orchestration (supervisor)
  plan_and_execute  guard → supervise (plans one task per specialist) → [specialist → decide → execute]
                    per task, in order → reply.
  react             guard → supervise ⇄ [specialist → decide → execute]: after each part the
                    supervisor sees the result and delegates the next part or finishes → reply.
strategy (specialists)
  plan_and_execute  plan the read and policy calls, run them, then propose.
  react             one tool call per turn, observing each result.
Orchestration defaults to `multi_agent.orchestration`, else to the specialist strategy.

An approver rejection sends a part back to its specialist once. Any node can divert to handoff,
which writes the packet and then goes to reply (hold reply). A part that needs a person or the
customer does not stop the other parts; at the end the strictest outcome decides the ticket
(human, then customer, then resolved).

LLMs decide what the customer needs, propose, review and write. The route tool, `TicketFlow` and `BackendWrites`
decide what executes: no token without a route at least as strict as the rule engine's
minimum, and no write without a token bound to that ticket, order, amount and items.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, TypedDict

import psycopg
from langgraph.graph import END, START, StateGraph

from support_ops.agents.approver import Approver
from support_ops.agents.guard import screen
from support_ops.agents.models import ApprovalResult, Case, Proposal, Task
from support_ops.agents.runtime import AgentContext, Budget, BudgetExceeded, Retries, llm_from_config
from support_ops.agents.specialist import STRATEGIES, Specialist
from support_ops.agents.toolbox import (ORDERS_SHIPPING_TOOLS, RETURNS_REFUNDS_TOOLS, Toolbox,
                                        returns_lookup_tool)
from support_ops.backend.postgres import PostgresReader, PostgresWriter
from support_ops.gateway.approvals import TicketFlow
from support_ops.gateway.route import RouteTool
from support_ops.gateway.tickets import ticket_fields
from support_ops.gateway.ticket_state import TicketState
from support_ops.gateway.writes import BackendWrites
from support_ops.helpers.cancellation import CancellationToken
from support_ops.helpers.clock import Clock
from support_ops.helpers.llm import ChatResult, LLM
from support_ops.helpers.tracing import Tracer
from support_ops.policy.provider import PolicyProvider
from support_ops.policy.routing import RoutingRules
from support_ops.policy.rules import Rules
from support_ops.supervisor.escalation import (FIXED_REPLY_TRIGGERS, HOLD_REPLY, confirm_question, customer_insists,
                                              handoff_packet, missing_records, plain_dashes)
from support_ops.supervisor.outcome import combined_outcome, customer_outcome
from support_ops.supervisor.supervisor import Supervisor
from support_ops.systems.single_agent import policy_from_config
from support_ops.systems.tools import SupportTools

SPECIALIST_TOOLS = {"orders_shipping": ORDERS_SHIPPING_TOOLS, "returns_refunds": RETURNS_REFUNDS_TOOLS}
DEFAULTS = {
    "orchestration": None,          # react or plan_and_execute; None: the specialists' strategy
    "max_orchestration_steps": 4,   # react: orchestrator decisions per ticket (delegations + finish)
    "policy_sections": {"orders_shipping": ["Orders and shipping"], "returns_refunds": ["Returns and refunds"]},
    # skills mode: each specialist's skill folders under policy.skills_dir
    "policy_skills": {"orders_shipping": ["orders-and-shipping"], "returns_refunds": ["returns-and-refunds"]},
}
# Earlier turns arrive with each request (the API is stateless).
HISTORY_ROLES = {"customer", "agent"}


class MultiAgentState(TypedDict, total=False):
    ticket_id: str
    customer_message: str
    history: list[dict[str, str]]
    strategy: str                   # the specialists' strategy
    orchestration: str              # the supervisor's: react or plan_and_execute
    next: str
    case: Case | None               # the supervisor's reading of the message: claims, not facts
    thoughts: list[str]             # the supervisor's reasoning at each decision
    redirect: bool                  # unrelated to the store: reply with what we can help with
    tasks: list[Task]               # from the supervisor: one per specialist, run in order
    part: int                       # index of the task in progress
    part_start: int                 # where the part's evidence starts
    parts: list[dict[str, Any]]     # finished parts: what each specialist proposed and how it ended
    orchestration_steps: int        # supervisor decisions made so far
    supervisor_note: str | None     # react: why code refused the last delegation
    proposal: Proposal | None
    min_route: dict[str, Any] | None
    approval: ApprovalResult | None
    feedback: list[str]
    reproposals: int
    evidence: list[dict[str, Any]]
    execution: dict[str, Any] | None
    customer_question: str | None
    unfound_record: str | None   # e.g. "order ORD-9": the customer confirmed it, the database lacks it
    handoff_reason: str | None
    handoff_packet: dict[str, Any] | None
    reply: str
    reply_review: ApprovalResult | None


class MultiAgentWorkflow:
    def __init__(self, llm: LLM, cfg: dict[str, Any], *, reader: PostgresReader | None = None,
                 policy: PolicyProvider | None = None, clock: Clock | None = None, tracer: Tracer | None = None,
                 cancellation: CancellationToken | None = None, writer: PostgresWriter | None = None):
        self.llm, self.cfg = llm, cfg
        self.settings = {**DEFAULTS, **cfg.get("multi_agent", {})}
        if moved := {"max_reproposals", "max_reply_drafts"} & set(self.settings):
            raise ValueError(f"multi_agent.{sorted(moved)} moved to review.max_rejections and "
                             "review.max_reply_drafts, shared with the single agent")
        # Shared with the single agent, so every arm gets the same second chances.
        review = cfg.get("review", {})
        self.settings["max_reproposals"] = review.get("max_rejections", 1)
        self.settings["max_reply_drafts"] = review.get("max_reply_drafts", 2)
        for key in ("policy_sections", "policy_skills"):
            # Per specialist, so a config naming one specialist's scope keeps the other's default scope
            # instead of leaving it unscoped (None, the whole policy, stays available when set explicitly).
            self.settings[key] = {**DEFAULTS[key], **(cfg.get("multi_agent", {}).get(key) or {})}
        self.retries = Retries.from_config(cfg)
        self.clock = clock or Clock(datetime.fromisoformat(cfg["run"]["clock_start"]))
        self._provided_tracer = tracer
        # Checked at every logged step (each LLM call, tool call and node), as in the single agent.
        self.cancellation = cancellation or CancellationToken()
        rules = Rules.load(cfg["policy"]["rules_file"])
        self.routing = RoutingRules.load(cfg["policy"]["rules_file"])
        self.reader = reader or PostgresReader()
        self.ticket_state = TicketState(self.clock, rules=rules)
        self.writer = writer = writer or PostgresWriter()
        self.tools = SupportTools(self.reader, policy or policy_from_config(cfg, rules), self.ticket_state, writer)
        policy = self.tools.policy
        if policy.strict_scope:   # e.g. a mistyped skill folder: fail now, not as empty lists mid-ticket
            for specialist, scope in self.settings[policy.scope_setting].items():
                if missing := policy.check_scope(scope):
                    raise ValueError(f"multi_agent.{policy.scope_setting} for {specialist} names parts the "
                                     f"policy does not have: {missing}")
        self.flow = TicketFlow(self.ticket_state, self.clock)
        self.writes = BackendWrites(self.flow, writer)
        self._tokens: dict[str, str | None] = {}   # kept out of graph state, so never returned
        graph = StateGraph(MultiAgentState)
        nodes = ("guard", "supervise", "specialist", "decide", "execute", "handoff", "reply")
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
               tracer: Tracer | None = None, history: list[dict[str, str]] | None = None,
               orchestration: str | None = None) -> dict[str, Any]:
        selected = strategy or self.cfg.get("agent", {}).get("strategy", "react")
        orchestration = orchestration or self.settings.get("orchestration") or selected
        for name, value in (("strategy", selected), ("orchestration", orchestration)):
            if value not in STRATEGIES:
                raise ValueError(f"{name} must be one of {', '.join(STRATEGIES)}")
        history = [{"role": turn["role"], "content": turn["content"]} for turn in history or []]
        if any(turn["role"] not in HISTORY_ROLES for turn in history):
            raise ValueError("history roles must be customer or agent")
        self.tracer = tracer or self._provided_tracer or Tracer(ticket_id)
        self.ctx = AgentContext(self._chat, self._event, self.clock, Budget(self.cfg["limits"], self.tracer),
                                self.retries)
        self.supervisor, self.approver = Supervisor(self.ctx), Approver(self.ctx)
        self._event("workflow_start", system="multi_agent", strategy=selected, orchestration=orchestration,
                    history_turns=len(history))
        state = self.graph.invoke({
            "ticket_id": ticket_id, "customer_message": customer_message, "history": history,
            "strategy": selected, "orchestration": orchestration, "case": None, "thoughts": [], "redirect": False, "proposal": None, "min_route": None, "approval": None,
            "feedback": [], "reproposals": 0, "evidence": [], "execution": None,
            "customer_question": None, "unfound_record": None, "handoff_reason": None, "handoff_packet": None,
            "tasks": [], "part": 0, "part_start": 0, "parts": [], "orchestration_steps": 0,
            "supervisor_note": None,
        }, {"recursion_limit": 40})
        status = self.flow.status(ticket_id)
        self._save_ticket(state)
        self._event("workflow_end", status=status, llm_calls=self.tracer.llm_calls,
                    elapsed_s=round(self.tracer.elapsed_s, 4))
        dump = lambda value: value.model_dump(mode="json") if value is not None else None
        return {
            "ticket_id": ticket_id, "strategy": selected, "orchestration": orchestration, "reply": state.get("reply", ""),
            "ticket_status": status, "handoff_packet": state.get("handoff_packet"),
            "case": dump(state.get("case")), "proposal": dump(state.get("proposal")),
            "min_route": state.get("min_route"), "approval": dump(state.get("approval")),
            "reply_review": dump(state.get("reply_review")), "execution": state.get("execution"),
            "evidence": state.get("evidence", []),
            "parts": [self._part_summary(part) for part in state.get("parts", [])],
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

    def _brief(self, state: MultiAgentState, part: bool = False) -> dict[str, Any]:
        """What an agent reads. With `part`, a multi-part ticket adds the part in progress."""
        brief: dict[str, Any] = {"message": plain_dashes(state["customer_message"])}
        tasks = state.get("tasks") or []
        if part and tasks and (len(tasks) > 1 or state.get("orchestration") == "react"):
            brief["your_task"] = plain_dashes(tasks[state["part"]].request)
        if state.get("history"):
            brief["history"] = [{**turn, "content": plain_dashes(turn["content"])} for turn in state["history"]]
        if state.get("case"):
            brief["case"] = state["case"].model_dump(mode="json")
        if state.get("feedback"):
            brief["approver_feedback"] = state["feedback"]
        return brief

    @staticmethod
    def _to_handoff(reason: str) -> dict[str, Any]:
        return {"next": "handoff", "handoff_reason": reason}

    # ------------------------------------------------------------- nodes

    def _guard(self, state: MultiAgentState) -> dict[str, Any]:
        self._event("node_start", node="guard", ticket_id=state["ticket_id"])
        flag = screen(self._chat, self.cfg, state["history"], state["customer_message"], self._event)
        return {"next": "supervise"} if flag == "benign" else self._to_handoff(f"guard: {flag}")

    def _supervise(self, state: MultiAgentState) -> dict[str, Any]:
        """The supervisor LLM decides what happens next; this only carries out its decision."""
        parts, steps, ticket_id = state["parts"], state["orchestration_steps"], state["ticket_id"]
        react, limit = state["orchestration"] == "react", self.settings["max_orchestration_steps"]
        if steps >= limit:
            self._event("orchestration_limit", steps=steps)
            return self._settle(state, {}) if parts else self._to_handoff(
                f"budget: the supervisor made {steps} decisions without a specialist result")
        brief = self._brief(state)
        if react:
            brief.update(steps_left=limit - steps, parts_done=[self._step_result(part) for part in parts],
                         earlier_thoughts=state["thoughts"])
        if state.get("supervisor_note"):
            brief["refused_last_step"] = state["supervisor_note"]
        update: dict[str, Any] = {"orchestration_steps": steps + 1, "supervisor_note": None}
        try:
            decision = self.supervisor.decide(brief, react=react, first=steps == 0)
        except ValueError as exc:
            return self._settle(state, update) if parts else {
                **update, **self._to_handoff(f"supervisor failed: {exc}")}
        update["thoughts"] = [*state["thoughts"], decision.thought]
        if decision.case:
            update["case"] = decision.case
        if decision.action == "handoff":
            # Parts already done stay in the packet and the reply.
            return {**update, **(self._last_part(parts) if parts else {}), **self._to_handoff(decision.reason)}
        if decision.action == "redirect" and not parts:
            # Unrelated to support: no specialist and no person; the reply says what we can help with.
            self._event("redirect", reason=decision.reason)
            self.flow.transition(ticket_id, "awaiting_customer", reason="off topic")
            return {**update, "redirect": True, "next": "reply"}
        if decision.action in {"finish", "redirect"}:
            if not parts:
                return {**update, **self._to_handoff("the supervisor finished before any specialist ran")}
            return self._settle(state, update)
        tasks = decision.tasks[:1] if react else decision.tasks
        if any(part["specialist"] == tasks[0].specialist and (part["execution"] or {}).get("status") == "executed"
               for part in parts):
            # One ticket needs one write per specialist; a second is more likely a repeat than a new need.
            note = f"{tasks[0].specialist} already executed a write on this ticket; it cannot be delegated again"
            self._event("supervisor_refused", specialist=tasks[0].specialist, reason=note)
            return {**update, "supervisor_note": note, "next": "supervise"}
        if self.flow.status(ticket_id) != "triaged":
            self.flow.transition(ticket_id, "triaged", reason="next part" if parts else decision.reason)
        return {**update, "tasks": [*state["tasks"], *tasks], "next": "specialist"}

    @staticmethod
    def _step_result(part: dict[str, Any]) -> dict[str, Any]:
        """What the supervisor sees of a finished part: enough to decide the next step."""
        proposal = part.get("proposal")
        return {"specialist": part["specialist"], "request": part["request"],
                "ended": {"approved": "resolved", "executed": "resolved"}.get(part["status"], part["status"]),
                "action": proposal.action if proposal else None, "finding": proposal.reason if proposal else None,
                "executed": (part.get("execution") or {}).get("status") == "executed",
                "customer_question": part.get("customer_question"), "handoff_reason": part.get("handoff_reason")}

    def _specialist(self, state: MultiAgentState) -> dict[str, Any]:
        task = state["tasks"][state["part"]]
        name = task.specialist
        if len(state["tasks"]) > 1:
            self._event("part_start", part=state["part"] + 1, of=len(state["tasks"]), specialist=name,
                        request=task.request)
        result = self._run_specialist(name, state["ticket_id"], self._brief(state, part=True), state["strategy"])
        update: dict[str, Any] = {
            "evidence": [*state["evidence"], *result.evidence],
            "proposal": result.proposal, "min_route": result.min_route,
        }
        if decided := self._missing_record_decision(state, result):
            return self._end_part(state, update, **decided)
        if result.error:
            reason = f"budget: {result.error}" if "budget" in result.error or "limit" in result.error \
                else f"specialist: {result.error}"
            return self._end_part(state, update, "awaiting_human", handoff_reason=reason)
        proposal = result.proposal
        # A question for the customer only ends the part when no person has to decide: a human route
        # (chosen, or forced by the rules) goes to a human even if the proposal also asks something.
        if proposal.action == "none" and proposal.customer_question and proposal.route != "human":
            return self._end_part(state, update, "awaiting_customer", customer_question=proposal.customer_question)
        self.flow.transition(state["ticket_id"], "proposed", route=proposal.route, operation=proposal.action)
        return {**update, "next": "decide"}

    def _run_specialist(self, name: str, ticket_id: str, brief: dict[str, Any], strategy: str):
        """Build the specialist's toolbox (its reads, its policy scope and the route tool) and run it.
        Shared with the pure-supervisor workflow so both arms' specialists are identical."""
        returns_checked: set[str] = set()
        orders_seen: list[str] = []
        route = RouteTool(self.reader, self.routing, self.clock, name, returns_checked, orders_seen)
        wrapped = {"policy", "get_returns_for_order"}
        reads = [tool for tool in SPECIALIST_TOOLS[name] if tool not in wrapped]
        policy = self.tools.policy
        scope = self.settings[policy.scope_setting].get(name)
        # A strict provider already refused a missing part at start-up; otherwise an empty search is never
        # a silent mystery.
        if not policy.strict_scope and (missing := policy.check_scope(scope)):
            self._event(f"{policy.scope_setting}_missing", specialist=name, missing=missing)
        extra: dict[str, Any] = {"policy": policy.scoped_tool(scope)}
        if "get_returns_for_order" in SPECIALIST_TOOLS[name]:
            extra["get_returns_for_order"] = returns_lookup_tool(self.tools, returns_checked)
        toolbox = Toolbox(self.tools, reads, {**extra, "route": route.as_tool()})
        return Specialist(name, toolbox, self.ctx, self.cfg, orders_seen).run(ticket_id, brief, strategy)

    def _missing_record_decision(self, state: MultiAgentState, result) -> dict[str, Any] | None:
        """A record the customer named was not found and the specialist has nothing else to do: ask the
        customer to check it, or, if they already gave the same ID before, hand it to a person."""
        missing = missing_records(result.evidence)
        proposal = result.proposal
        nothing_else = result.error is not None or (proposal is not None and proposal.action == "none")
        if not missing or not nothing_else:
            return None
        message = state["customer_message"]
        insisted = [(name, record_id) for name, record_id in missing
                    if customer_insists(state["history"], message, record_id)]
        if insisted:
            name, record_id = insisted[0]
            return {"status": "awaiting_human", "unfound_record": f"{name} {record_id}",
                    "handoff_reason": f"the customer confirmed {name} {record_id}, but it cannot be found"}
        if proposal is not None and proposal.customer_question and proposal.route != "human":
            return None   # the specialist is already asking the customer
        question = (proposal.customer_question if proposal and proposal.customer_question
                    else confirm_question(missing, message))
        self._event("missing_record", records=[f"{name} {record_id}" for name, record_id in missing], question=question)
        return {"status": "awaiting_customer", "customer_question": question}

    def _end_part(self, state: MultiAgentState, update: dict[str, Any], status: str, *,
                  handoff_reason: str | None = None, customer_question: str | None = None,
                  unfound_record: str | None = None) -> dict[str, Any]:
        """Record how this part ended (`status`: approved, executed, awaiting_customer or awaiting_human),
        then start the next part, or settle the ticket on the strictest outcome of all parts."""
        merged, ticket_id = {**state, **update}, state["ticket_id"]
        task = state["tasks"][state["part"]]
        part = {"specialist": task.specialist, "request": task.request, "status": status,
                **{key: merged.get(key) for key in ("proposal", "min_route", "approval", "execution")},
                "customer_question": customer_question, "handoff_reason": handoff_reason,
                "unfound_record": unfound_record, "evidence": merged["evidence"][state["part_start"]:]}
        parts = [*state["parts"], part]
        update = {**update, "parts": parts}
        self._save_ticket({**merged, "parts": parts})
        react = state["orchestration"] == "react"
        if react or state["part"] + 1 < len(state["tasks"]):
            self._event("part_done", part=state["part"] + 1, specialist=task.specialist, status=status)
            # Under react the supervisor may finish instead, so the ticket keeps this part's state
            # until it delegates again.
            if not react and self.flow.status(ticket_id) != "triaged":
                self.flow.transition(ticket_id, "triaged", reason="next part")
            return {**update, "part": state["part"] + 1, "part_start": len(merged["evidence"]),
                    "proposal": None, "min_route": None, "approval": None, "execution": None,
                    "feedback": [], "reproposals": 0, "next": "supervise" if react else "specialist"}
        return self._settle(state, update)

    def _settle(self, state: MultiAgentState, update: dict[str, Any]) -> dict[str, Any]:
        """All parts are done: the strictest outcome decides the ticket (human, customer, resolved).
        Under react the last part's proposal and results are restored from its record."""
        parts, ticket_id = update.get("parts", state["parts"]), state["ticket_id"]
        update = {**update, **self._last_part(parts)}
        if humans := [done for done in parts if done["status"] == "awaiting_human"]:
            reason = humans[0]["handoff_reason"] if len(parts) == 1 else "; ".join(
                f"{done['specialist']}: {done['handoff_reason']}" for done in humans)
            unfound = next((done["unfound_record"] for done in humans if done["unfound_record"]), None)
            return {**update, "unfound_record": unfound, **self._to_handoff(reason)}
        if asking := [done for done in parts if done["status"] == "awaiting_customer"]:
            self.flow.transition(ticket_id, "awaiting_customer")
            return {**update, "customer_question": " ".join(done["customer_question"] for done in asking),
                    "next": "reply"}
        return {**update, "next": "reply"}

    def _save_ticket(self, state: MultiAgentState) -> None:
        """Keep `support_tickets` current. A failed save is logged: it must not cost the customer their reply."""
        ticket_id = state["ticket_id"]
        try:
            self.writer.save_ticket(ticket_id=ticket_id, at=self.clock.now(), **ticket_fields(
                flow_status=self.flow.status(ticket_id), case=state.get("case"),
                customer_message=state["customer_message"], parts=state.get("parts", []),
                proposal=state.get("proposal"), evidence=state.get("evidence", []), reply=state.get("reply"),
                handoff_packet=state.get("handoff_packet")))
        except (psycopg.Error, RuntimeError) as exc:
            self.tracer.log("ticket_save_failed", ticket_id=ticket_id, error=str(exc)[:300])

    @staticmethod
    def _last_part(parts: list[dict[str, Any]]) -> dict[str, Any]:
        return {key: parts[-1][key] for key in ("proposal", "min_route", "approval", "execution")}

    @staticmethod
    def _part_summary(part: dict[str, Any]) -> dict[str, Any]:
        """A finished part for the response and the handoff packet (its evidence is in the ticket's)."""
        dump = lambda value: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        return {key: dump(value) for key, value in part.items() if key != "evidence"}

    def _decide(self, state: MultiAgentState) -> dict[str, Any]:
        self._event("node_start", node="decide", route=state["proposal"].route)
        ticket_id, proposal, minimum = state["ticket_id"], state["proposal"], state["min_route"]
        if proposal.route == "human":
            reasons = "; ".join(minimum.get("reasons", [])) if minimum else ""
            return self._end_part(state, {}, "awaiting_human",
                                  handoff_reason="route: human" + (f" ({reasons})" if reasons else ""))
        approval = None
        if proposal.route == "approver":
            approval = self.approver.check_proposal(self._brief(state, part=True), proposal, minimum,
                                                    state["evidence"][state["part_start"]:])
            if approval.decision == "reject" and state["reproposals"] < self.settings["max_reproposals"]:
                self.flow.transition(ticket_id, "triaged", reason="approver rejected the proposal")
                return {"approval": approval, "feedback": approval.issues,
                        "reproposals": state["reproposals"] + 1, "next": "specialist"}
            if approval.decision != "approve":
                return self._end_part(state, {"approval": approval}, "awaiting_human",
                                      handoff_reason=f"approver {approval.decision}: {'; '.join(approval.issues)}")
        try:
            granted = self.flow.approve(ticket_id, proposal, minimum, approval)
        except ValueError as exc:
            return self._end_part(state, {"approval": approval}, "awaiting_human",
                                  handoff_reason=f"gateway refused: {exc}")
        if not granted["approval_token"]:   # information only: nothing to execute
            return self._end_part(state, {"approval": approval}, "approved")
        self._tokens[ticket_id] = granted["approval_token"]
        return {"approval": approval, "next": "execute"}

    def _execute(self, state: MultiAgentState) -> dict[str, Any]:
        self._event("node_start", node="execute")
        ticket_id, proposal = state["ticket_id"], state["proposal"]
        lines = sorted(f"{line.item_id}x{line.quantity}" for line in proposal.items)
        result = self.writes.execute(
            ticket_id, token=self._tokens.pop(ticket_id, None) or "", operation=proposal.action,
            order_id=proposal.order_id, amount=proposal.amount, items=[line.model_dump() for line in proposal.items],
            reason=proposal.reason_code or proposal.reason,
            idempotency_key=f"{ticket_id}:{proposal.action}:{proposal.order_id}:{proposal.amount}:{','.join(lines)}")
        self._event("write_result", result=result)
        if result["status"] != "executed":
            return self._end_part(state, {"execution": result}, "awaiting_human",
                                  handoff_reason=f"write refused: {result.get('reason')}")
        return self._end_part(state, {"execution": result}, "executed")

    def _handoff(self, state: MultiAgentState) -> dict[str, Any]:
        self._event("node_start", node="handoff", reason=state["handoff_reason"])
        packet = handoff_packet(
            ticket_id=state["ticket_id"], reason=state["handoff_reason"], customer_message=state["customer_message"],
            case=state.get("case"), proposal=state.get("proposal"), min_route=state.get("min_route"),
            evidence=state["evidence"], execution=state.get("execution"),
            parts=[self._part_summary(part) for part in state.get("parts", [])] if len(state.get("parts", [])) > 1 else [])
        if self.flow.status(state["ticket_id"]) != "awaiting_human":
            self.flow.handoff(packet)
        self._event("handoff_created", packet=packet.model_dump(mode="json"))
        return {"handoff_packet": packet.model_dump(mode="json"), "next": "reply"}

    def _reply(self, state: MultiAgentState) -> dict[str, Any]:
        ticket_id = state["ticket_id"]
        reason = state.get("handoff_reason") or ""
        if reason.startswith(FIXED_REPLY_TRIGGERS):
            return {"reply": HOLD_REPLY}
        status = self.flow.status(ticket_id)
        parts = state.get("parts", [])
        if len(parts) > 1:
            outcome = combined_outcome(status, [
                {"customer_request": part["request"], **customer_outcome(
                    status=part["status"], evidence=part["evidence"], proposal=part["proposal"],
                    execution=part["execution"], min_route=part["min_route"],
                    customer_question=part["customer_question"], currency=self.routing.currency,
                    unfound_record=part["unfound_record"])}
                for part in parts])
        else:
            outcome = customer_outcome(
                status=status, evidence=state["evidence"], proposal=state.get("proposal"),
                execution=state.get("execution"), min_route=state.get("min_route"),
                customer_question=state.get("customer_question"), currency=self.routing.currency,
                redirect=bool(state.get("redirect")),
                unfound_record=state.get("unfound_record"))
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
