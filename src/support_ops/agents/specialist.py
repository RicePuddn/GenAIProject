"""Specialist agents: gather facts, read policy, and finish by calling `route` with a proposal.

One class serves both domains; the domain sets the toolbox and the scope in the prompt.
The inner loop follows `agent.strategy`:

react             one tool call per turn, observing each result (as in the single agent)
plan_and_execute  plan the read and policy calls, run them, then write the proposal; replan on errors
reflexion         Reflexion (Shinn et al., 2023): react attempts ("trials"), each with a fresh
                  conversation. A failed trial (no valid proposal here, or the approver rejecting it in
                  the graph) is followed by a verbal self-reflection, kept in the ticket's long-term
                  memory; the next trial sees the latest `multi_agent.reflection_window` lessons. Up to
                  `multi_agent.max_trials` trials. The memory lasts for the ticket, across chat turns.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from support_ops.agents.models import SPECIALIST_ACTIONS, Proposal, Specialist as SpecialistName
from support_ops.agents.runtime import AgentContext, BudgetExceeded, parse_json

REFLECT_PROMPT = (
    "You reflect on a failed attempt by a support specialist, so the next attempt does better. In 2-3 "
    "sentences, diagnose the specific mistake that caused the failure and state a concrete plan for the "
    "next attempt: which facts to check, which clause applies, which action, amounts, items or route. "
    "Do not restate the case and do not write to the customer."
)
# Failures a reflection cannot fix, such as a database outage, end the trials.
UNREFLECTABLE = ("tool execution failed",)
from support_ops.agents.toolbox import Toolbox
from support_ops.helpers.llm import ChatResult, ToolCall

STRATEGIES = ("react", "plan_and_execute", "reflexion")

SCOPES: dict[SpecialistName, str] = {
    "orders_shipping": "order status, shipment tracking, delivery delays and order changes "
                       "(cancellation or address change: action order_change)",
    "returns_refunds": "return eligibility and return labels (action return), exchanges (action exchange), "
                       "both with items: the order item IDs and quantities coming back, from the order's "
                       "items; and refunds (action refund, with amount and reason_code)",
}


@dataclass
class SpecialistResult:
    proposal: Proposal | None = None
    min_route: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)   # lessons written during this run
    error: str | None = None
    orders_seen: list[str] = field(default_factory=list)   # order IDs the read tools were called with


class Specialist:
    def __init__(self, name: SpecialistName, toolbox: Toolbox, ctx: AgentContext, cfg: dict[str, Any]):
        self.name, self.toolbox, self.ctx, self.cfg = name, toolbox, ctx, cfg
        settings = cfg.get("multi_agent", {})
        self.max_turns = cfg["limits"]["max_agent_steps"]
        self.max_replans = cfg["limits"].get("max_replans", 2)
        self.max_trials = settings.get("max_trials", 3)
        self.reflection_window = settings.get("reflection_window", 3)

    def system_prompt(self) -> str:
        return (
            f"You are the {self.name.replace('_', ' ')} specialist in a policy-bound support team. "
            f"Current workflow date: {self.ctx.clock.now().date()}. Your scope: {SCOPES[self.name]}. "
            f"Actions you may propose: {', '.join(sorted(SPECIALIST_ACTIONS[self.name]))}; for anything else, "
            "propose action none with route human. Policy search covers your own policy sections; search "
            "with scope all when the case crosses into another area. "
            "Use the read tools for account facts; read an order by its own ID and use the order lookup "
            "tools for its items, payments, returns and shipments. "
            f"{self.toolbox.policy.prompt_hint} "
            "Calculate return windows from the workflow date and the delivery date. "
            "You only propose: finish by calling the route tool once with your proposal, citing the "
            "clause IDs or policy files you read. Choose route auto only for low-value, low-risk cases; "
            "approver for a second check; human when an operator must decide. If the case is not what "
            "the cited clause covers (for example a defective or used item when the clause covers unused "
            "items), set policy_exception true or route human; never stretch a clause to fit. If only the "
            "customer can supply a missing fact, call route with action none, route auto and customer_question. "
            "Never write to the customer yourself: the supervisor writes the reply from your proposal. "
            "If reflections_from_earlier_attempts are given, they are your own lessons from failed attempts "
            "at this ticket; follow them. Never invent facts, amounts or policy terms."
        )

    def run(self, ticket_id: str, brief: dict[str, Any], strategy: str,
            memory: list[str] | None = None) -> SpecialistResult:
        """One specialist run. `memory` is the ticket's reflections so far (reflexion only)."""
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}")
        result = SpecialistResult()
        self.ctx.log("node_start", node="specialist", specialist=self.name, strategy=strategy)
        try:
            if strategy == "plan_and_execute":
                self._plan_and_execute(ticket_id, brief, result)
            elif strategy == "reflexion":
                self._reflexion(ticket_id, brief, result, list(memory or []))
            else:
                self._react(ticket_id, brief, result)
        except BudgetExceeded as exc:
            result.error = str(exc)
        if result.proposal is None and result.error is None:
            result.error = "specialist did not submit a proposal"
        self.ctx.log("specialist_end", specialist=self.name, error=result.error,
                     proposal=result.proposal.model_dump(mode="json") if result.proposal else None)
        return result

    # ------------------------------------------------------------- tools

    def _execute(self, ticket_id: str, call_id: str, name: str, arguments: Any,
                 result: SpecialistResult) -> dict[str, Any]:
        self.ctx.budget.check()
        seen = sorted(set(result.orders_seen))
        if name == "route" and isinstance(arguments, dict) and not arguments.get("order_id") and len(seen) == 1:
            # The case is about the one order the specialist looked up, so the rule engine checks that
            # order (e.g. for a late shipment) even when an information-only proposal leaves it out.
            arguments = {**arguments, "order_id": seen[0]}
            self.ctx.log("order_id_filled", call_id=call_id, order_id=seen[0], agent=self.name)
        self.ctx.log("tool_call_requested", call_id=call_id, tool=name, input=arguments, agent=self.name)
        self.ctx.log("tool_start", call_id=call_id, tool=name, input=arguments, agent=self.name)
        started = time.perf_counter()
        attempts = self.ctx.retries.tool_errors + 1
        for attempt in range(1, attempts + 1):
            try:
                value = self.toolbox.execute(ticket_id, name, arguments)
            except (ValueError, FileNotFoundError) as exc:
                # The agent's own mistake: the same arguments would fail again, so it gets the error.
                value = {"error": str(exc)}
            except Exception as exc:   # e.g. a database timeout: worth running again as is
                value = {"error": str(exc)}
                if attempt < attempts:
                    self.ctx.log("tool_retry", call_id=call_id, tool=name, attempt=attempt, error=str(exc))
                    time.sleep(self.ctx.retries.backoff_s * attempt)
                    continue
                result.error = f"tool execution failed after {attempts} attempt(s): {exc}"
            break
        self.ctx.budget.tool_steps += 1
        self.ctx.log("tool_end", call_id=call_id, tool=name, agent=self.name,
                     elapsed_s=round(time.perf_counter() - started, 6), output=value)
        result.evidence.append({"tool": name, "agent": self.name, "result": value})
        if name != "route" and "error" not in value and isinstance(arguments, dict):
            order_id = arguments.get("order_id") or (
                arguments.get("record_id") if arguments.get("kind") == "orders" else None)
            if order_id:
                result.orders_seen.append(order_id)
        if name == "route" and value.get("status") == "recorded":
            result.proposal = Proposal.model_validate(value["proposal"])
            result.min_route = value["min_route"]
        return value

    def _user_message(self, brief: dict[str, Any]) -> dict[str, str]:
        return {"role": "user", "content": json.dumps(brief, default=str)}

    # ------------------------------------------------------------- react

    def _react(self, ticket_id: str, brief: dict[str, Any], result: SpecialistResult) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt() +
             " Choose exactly one tool call per turn."},
            self._user_message(brief),
        ]
        nudged = False
        for _ in range(self.max_turns):
            self.ctx.budget.check()
            reply = self.ctx.chat("specialist", messages, tools=self.toolbox.schemas)
            if not reply.tool_calls:
                if nudged:
                    return
                nudged = True
                messages += [{"role": "assistant", "content": reply.content or ""},
                             {"role": "user", "content": "Finish by calling the route tool with your proposal."}]
                continue
            chosen = reply.tool_calls[0]
            if len(reply.tool_calls) > 1:
                self.ctx.log("tool_calls_deferred", tools=[call.name for call in reply.tool_calls[1:]])
            call_id = self.ctx.next_call_id()
            messages.append(ChatResult(None, [ToolCall(call_id, chosen.name, chosen.arguments)]).as_message())
            value = self._execute(ticket_id, call_id, chosen.name, chosen.arguments, result)
            messages.append(self._tool_message(call_id, chosen.name, value))
            if result.proposal is not None or result.error is not None:
                return
        result.error = result.error or "specialist turn limit reached"

    @staticmethod
    def _tool_message(call_id: str, name: str, value: dict[str, Any]) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": call_id, "name": name, "content": json.dumps(value, default=str)}

    # ------------------------------------------------------------- reflexion

    def with_memory(self, brief: dict[str, Any], memory: list[str]) -> dict[str, Any]:
        """The brief for a new trial: the latest lessons, never the earlier trials' conversations."""
        recent = memory[-self.reflection_window:] if self.reflection_window else []
        return {**brief, "reflections_from_earlier_attempts": recent} if recent else brief

    def _reflexion(self, ticket_id: str, brief: dict[str, Any], result: SpecialistResult, memory: list[str]) -> None:
        for trial in range(1, self.max_trials + 1):
            self.ctx.log("reflexion_trial", specialist=self.name, trial=trial,
                         memory=memory[-self.reflection_window:] if self.reflection_window else [])
            # A fresh attempt: its own conversation and result; facts it gathers still count for the reply.
            attempt = SpecialistResult(orders_seen=list(result.orders_seen))
            try:
                self._react(ticket_id, self.with_memory(brief, memory), attempt)
            finally:
                result.evidence += attempt.evidence
                result.orders_seen = attempt.orders_seen
            if attempt.proposal is not None:
                result.proposal, result.min_route = attempt.proposal, attempt.min_route
                return
            failure = attempt.error or "the attempt ended without a valid proposal"
            if failure.startswith(UNREFLECTABLE) or trial == self.max_trials:
                result.error = failure if failure.startswith(UNREFLECTABLE) \
                    else f"no valid proposal after {trial} trials: {failure}"
                return
            lesson = self.reflect(brief, attempt.evidence, None, failure, memory, trial)
            memory.append(lesson)
            result.reflections.append(lesson)

    def reflect(self, brief: dict[str, Any], evidence: list[dict[str, Any]], proposal: Proposal | None,
                failure: str, memory: list[str], trial: int) -> str:
        """Self-reflection on a failed trial: a short lesson for the next one."""
        self.ctx.budget.check()
        reply = self.ctx.chat("specialist", [
            {"role": "system", "content": REFLECT_PROMPT},
            {"role": "user", "content": json.dumps({
                "case": {key: value for key, value in brief.items() if key != "reflections_from_earlier_attempts"},
                "attempt": {"tool_calls_and_results": evidence,
                            "proposal": proposal.model_dump(mode="json") if proposal else None},
                "why_it_failed": failure,
                "earlier_reflections": memory[-self.reflection_window:] if self.reflection_window else [],
            }, default=str)},
        ])
        lesson = (reply.content or "").strip() or f"The last attempt failed because {failure}; fix that first."
        self.ctx.log("reflection", specialist=self.name, trial=trial, failure=failure, reflection=lesson)
        return lesson

    # ------------------------------------------------------------- plan and execute

    def _plan_and_execute(self, ticket_id: str, brief: dict[str, Any], result: SpecialistResult) -> None:
        reads = [schema for schema in self.toolbox.schemas if schema["function"]["name"] != "route"]
        policy_context = self.toolbox.policy.planner_context()
        previous_error: str | None = None
        for attempt in range(self.max_replans + 1):
            self.ctx.budget.check()
            reply = self.ctx.chat("specialist", [
                {"role": "system", "content": self.system_prompt() +
                 ' First plan the read and policy calls you need, in order. Return JSON only as '
                 '{"calls":[{"name":"EXACT_TOOL_NAME","arguments":{}}]}. Use only these tool schemas: '
                 + json.dumps(reads) + ". " + (policy_context + ". " if policy_context else "") +
                 "Use literal known IDs; never invent results of earlier calls. Do not include route; "
                 "you will write the proposal after the calls run. Correct any previous error using the evidence."},
                {"role": "user", "content": json.dumps({
                    **brief, "evidence": result.evidence, "previous_error": previous_error}, default=str)},
            ], json_mode=True)
            try:
                calls = self._validated_plan(reply.content)
            except ValueError as exc:
                previous_error = f"invalid plan: {exc}"
                self.ctx.log("plan_rejected", reason=previous_error, replan=attempt + 1)
                continue
            self.ctx.log("plan", specialist=self.name, calls=calls, attempt=attempt + 1)
            for name, arguments in calls:
                self._execute(ticket_id, self.ctx.next_call_id(), name, arguments, result)
                if result.error:
                    return
            self.ctx.budget.check()
            reply = self.ctx.chat("specialist", [
                {"role": "system", "content": self.system_prompt() +
                 " The planned calls have run. Write your proposal as JSON only, matching the route tool's "
                 "arguments: " + json.dumps(self.toolbox.argument_models["route"].model_json_schema())},
                {"role": "user", "content": json.dumps({
                    **brief, "evidence": result.evidence, "previous_error": previous_error}, default=str)},
            ], json_mode=True)
            try:
                draft = parse_json(reply.content)
            except ValueError as exc:
                previous_error = str(exc)
                continue
            value = self._execute(ticket_id, self.ctx.next_call_id(), "route", draft, result)
            if result.proposal is not None or result.error:
                return
            previous_error = value.get("error")
            self.ctx.log("plan_rejected", reason=previous_error, replan=attempt + 1)
        result.error = f"no valid proposal after {self.max_replans + 1} plans: {previous_error}"

    def _validated_plan(self, content: str | None) -> list[tuple[str, dict[str, Any]]]:
        plan = parse_json(content)
        if not isinstance(plan, dict) or not isinstance(plan.get("calls"), list):
            raise ValueError("plan must contain a calls list")
        remaining = self.cfg["limits"]["max_steps_per_ticket"] - self.ctx.budget.tool_steps
        if len(plan["calls"]) >= remaining:
            raise ValueError("plan exceeds the remaining tool budget")
        calls = []
        for call in plan["calls"]:
            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                raise ValueError("each call requires exactly name and arguments")
            if call["name"] == "route":
                raise ValueError("route is not part of the plan")
            calls.append((call["name"], self.toolbox.validate(call["name"], call["arguments"])))
        return calls
