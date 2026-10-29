"""Specialist agents: gather facts, read policy, and finish by calling `route` with a proposal.

One class serves both domains; the domain sets the toolbox and the scope in the prompt.
The inner loop follows `agent.strategy`:

react             one tool call per turn, observing each result (as in the single agent)
plan_and_execute  plan the read and policy calls, run them, then write the proposal; replan on errors
reflexion         react, but before submitting a proposal the specialist critiques it against the
                  evidence and revises it (up to `multi_agent.max_reflections` times)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from support_ops.agents.models import SPECIALIST_ACTIONS, Proposal, Specialist as SpecialistName
from support_ops.agents.runtime import AgentContext, BudgetExceeded, parse_json
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
    reflections: list[str] = field(default_factory=list)
    error: str | None = None
    orders_seen: list[str] = field(default_factory=list)   # order IDs the read tools were called with


class Specialist:
    def __init__(self, name: SpecialistName, toolbox: Toolbox, ctx: AgentContext, cfg: dict[str, Any]):
        self.name, self.toolbox, self.ctx, self.cfg = name, toolbox, ctx, cfg
        settings = cfg.get("multi_agent", {})
        self.max_turns = cfg["limits"]["max_agent_steps"]
        self.max_replans = cfg["limits"].get("max_replans", 2)
        self.max_reflections = settings.get("max_reflections", 2)

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
            "Never invent facts, amounts or policy terms."
        )

    def run(self, ticket_id: str, brief: dict[str, Any], strategy: str) -> SpecialistResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}")
        result = SpecialistResult()
        self.ctx.log("node_start", node="specialist", specialist=self.name, strategy=strategy)
        try:
            if strategy == "plan_and_execute":
                self._plan_and_execute(ticket_id, brief, result)
            else:
                self._react(ticket_id, brief, result, reflect=strategy == "reflexion")
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

    # ------------------------------------------------------------- react / reflexion

    def _react(self, ticket_id: str, brief: dict[str, Any], result: SpecialistResult, reflect: bool) -> None:
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
            if reflect and chosen.name == "route" and len(result.reflections) < self.max_reflections:
                critique = self._critique(brief, result, chosen.arguments)
                if critique:
                    result.reflections.append(critique)
                    # The proposal is held back; the critique is what the specialist observes instead.
                    messages.append(self._tool_message(call_id, "route", {
                        "status": "not_submitted", "self_review": critique,
                        "instruction": "Fix the issues, gathering facts if needed, then call route again."}))
                    continue
            value = self._execute(ticket_id, call_id, chosen.name, chosen.arguments, result)
            messages.append(self._tool_message(call_id, chosen.name, value))
            if result.proposal is not None or result.error is not None:
                return
        result.error = result.error or "specialist turn limit reached"

    @staticmethod
    def _tool_message(call_id: str, name: str, value: dict[str, Any]) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": call_id, "name": name, "content": json.dumps(value, default=str)}

    def _critique(self, brief: dict[str, Any], result: SpecialistResult, arguments: Any) -> str | None:
        """Self-review of a draft proposal; returns the critique, or None when it holds up."""
        try:
            self.toolbox.validate("route", arguments)
        except ValueError:
            return None   # malformed drafts get the tool's own validation error instead
        self.ctx.budget.check()
        reply = self.ctx.chat("specialist", [
            {"role": "system", "content":
                "You review a support specialist's draft proposal before it is submitted. Check that every "
                "fact (order, items, amounts paid, delivery date) was confirmed by a tool result; that the "
                "amount does not exceed what was paid for the affected items; that the cited clauses were "
                "actually read and apply, including any exception to them; and that the route is strict "
                "enough. Return JSON only: {\"ok\": true|false, \"critique\": \"specific problems, or empty\"}."},
            {"role": "user", "content": json.dumps({
                "case": brief, "evidence": result.evidence, "draft_proposal": arguments,
                "earlier_critiques": result.reflections,
            }, default=str)},
        ], json_mode=True)
        try:
            verdict = parse_json(reply.content)
            ok, critique = bool(verdict.get("ok", True)), str(verdict.get("critique") or "").strip()
        except (ValueError, AttributeError):
            self.ctx.log("reflection_unparseable", response=reply.content)
            return None
        self.ctx.log("reflection", specialist=self.name, ok=ok, critique=critique,
                     round=len(result.reflections) + 1)
        return None if ok or not critique else critique

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
