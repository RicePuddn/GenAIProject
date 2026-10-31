"""Specialist agents: gather facts, read policy, and finish by calling `route` with a proposal.

One class serves both domains; the domain sets the toolbox and the scope in the prompt.
The inner loop follows `agent.strategy`:

react             one tool call per turn, observing each result (as in the single agent)
plan_and_execute  plan the read and policy calls, run them, then write the proposal; replan on errors
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from support_ops.agents.models import SPECIALIST_ACTIONS, Proposal, ProposalDraft, Specialist as SpecialistName
from support_ops.agents.runtime import AgentContext, BudgetExceeded, chat_json, parse_json
from support_ops.agents.toolbox import Toolbox
from support_ops.helpers.llm import ChatResult, ToolCall

# The same tool failing with the same error this many times in a row ends the attempt.
MAX_REPEATED_ERRORS = 3

STRATEGIES = ("react", "plan_and_execute")

SCOPES: dict[SpecialistName, str] = {
    "orders_shipping": "questions about an order itself (its items, quantities, prices and status), "
                       "shipment tracking, delivery delays and order changes (action order_change: reason_code "
                       "cancel for a cancellation; other changes go to a human). You cannot see payments, so "
                       "never say whether an order was paid; any refund after a cancellation is handled "
                       "separately by the returns and refunds specialist",
    "returns_refunds": "return eligibility and return labels (action return), exchanges (action exchange), "
                       "both with items: the order item IDs and quantities coming back, from the order's "
                       "items; and refunds (action refund, with amount, reason_code and the items it is "
                       "for; never more than was paid for them). For a cancelled order with no return, check its payments: refund "
                       "the captured balance less earlier and pending refunds, listing no items; if nothing "
                       "is owed, propose action none",
}


def citation_forms(clause: str) -> set[str]:
    """The ways a citation may name what was read, casefolded: as written, each side of a `#` anchor, a
    path cut to its last two parts (skill/file, so another skill's file of the same name does not match),
    and the leading clause ID of "RET-1 Return window"."""
    forms = set()
    for part in clause.strip().casefold().split("#"):
        part = part.strip().strip("/")
        if not part:
            continue
        pieces = part.split("/")
        forms |= {part, "/".join(pieces[-2:]), part.split()[0]}
    return forms


@dataclass
class SpecialistResult:
    proposal: Proposal | None = None
    min_route: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    orders_seen: list[str] = field(default_factory=list)   # order IDs the read tools were called with
    citable: set[str] = field(default_factory=set)   # clause IDs and files the policy tool returned


class Specialist:
    def __init__(self, name: SpecialistName, toolbox: Toolbox, ctx: AgentContext, cfg: dict[str, Any],
                 orders_seen: list[str] | None = None):
        self.name, self.toolbox, self.ctx, self.cfg = name, toolbox, ctx, cfg
        # Shared with the route tool, which checks every order looked up when a proposal names none.
        self.orders_seen = orders_seen if orders_seen is not None else []
        self.max_turns = cfg["limits"]["max_agent_steps"]
        self.max_replans = cfg["limits"].get("max_replans", 2)

    def system_prompt(self) -> str:
        return (
            f"You are the {self.name.replace('_', ' ')} specialist in a policy-bound support team. "
            f"Current workflow date: {self.ctx.clock.now().date()}. Your scope: {SCOPES[self.name]}. "
            f"Actions you may propose: {', '.join(sorted(SPECIALIST_ACTIONS[self.name]))}; for anything else, "
            f"propose action none with route human. {getattr(self.toolbox.policy, 'scope_hint', '')} "
            "Use the read tools for account facts; read an order by its own ID and use the order lookup "
            "tools for its items, payments, returns and shipments. Before advising on a return or refund, "
            "check the order's items and any existing returns or refunds. "
            f"{self.toolbox.policy.prompt_hint} "
            "Calculate return windows from the workflow date and the delivery date. "
            "You only propose: finish by calling the route tool once with your proposal, citing the "
            "clause IDs or policy files you read. Choose route auto only for low-value, low-risk cases; "
            "approver for a second check; human when an operator must decide. If the case is not what "
            "the cited clause covers (for example a defective or used item when the clause covers unused "
            "items), set policy_exception true or route human; never stretch a clause to fit. If only the "
            "customer can supply a missing fact, call route with action none, route auto and customer_question. "
            "If an order or other record the customer named is not found, ask them to confirm it the same way; "
            "do not route it to a human. "
            "If the brief has your_task, handle only that part of the message: another specialist handles "
            "the rest, so do not propose or ask about it. "
            "Never write to the customer yourself: the supervisor writes the reply from your proposal. "
            "Never invent facts, amounts or policy terms."
        )

    def run(self, ticket_id: str, brief: dict[str, Any], strategy: str) -> SpecialistResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}")
        result = SpecialistResult(orders_seen=self.orders_seen)
        self.ctx.log("node_start", node="specialist", specialist=self.name, strategy=strategy)
        try:
            if strategy == "plan_and_execute":
                self._plan_and_execute(ticket_id, brief, result)
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
        unnamed = name == "route" and isinstance(arguments, dict) and not arguments.get("order_id")
        if unnamed and len(seen) == 1:
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
                if name == "route":
                    self._check_citations(arguments, result)
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
        if name == "policy" and "error" not in value:
            result.citable |= getattr(self.toolbox.policy, "citable", lambda _: set())(value)
        if name != "route" and "error" not in value and isinstance(arguments, dict):
            order_id = arguments.get("order_id") or (
                arguments.get("record_id") if arguments.get("kind") == "orders" else None)
            if order_id:
                result.orders_seen.append(order_id)
        if name == "route" and value.get("status") == "recorded":
            result.proposal = Proposal.model_validate(value["proposal"])
            result.min_route = value["min_route"]
        return value

    def _check_citations(self, arguments: Any, result: SpecialistResult) -> None:
        """A proposal may cite only clauses or files its policy tool returned in this case."""
        if not self.cfg.get("multi_agent", {}).get("check_citations", True) or not isinstance(arguments, dict):
            return
        cited = arguments.get("cited_clauses") or []
        citable = {item.casefold() for item in result.citable}
        if unread := [str(clause) for clause in cited if not citation_forms(str(clause)) & citable]:
            self.ctx.log("citation_rejected", agent=self.name, unread=unread)
            raise ValueError(f"cited_clauses {unread} were not returned by your policy tool in this case; "
                             "read them first, or cite only the clause IDs or files you read")

    def _user_message(self, brief: dict[str, Any]) -> dict[str, str]:
        return {"role": "user", "content": json.dumps(brief, default=str)}

    # ------------------------------------------------------------- react

    def _react(self, ticket_id: str, brief: dict[str, Any], result: SpecialistResult) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt() +
             " Choose exactly one tool call per turn."},
            self._user_message(brief),
        ]
        nudged, last_error, repeats = False, None, 0
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
            error = (chosen.name, value["error"]) if "error" in value else None
            repeats = repeats + 1 if error and error == last_error else (1 if error else 0)
            last_error = error
            if repeats >= MAX_REPEATED_ERRORS:
                # Retrying something that keeps failing the same way only spends turns and tokens.
                self.ctx.log("repeated_error", tool=chosen.name, error=value["error"], times=repeats)
                result.error = f"{chosen.name} failed {repeats} times in a row with: {value['error']}"
                return
        result.error = result.error or "specialist turn limit reached"

    @staticmethod
    def _tool_message(call_id: str, name: str, value: dict[str, Any]) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": call_id, "name": name, "content": json.dumps(value, default=str)}

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
            failed = None
            for name, arguments in calls:
                value = self._execute(ticket_id, self.ctx.next_call_id(), name, arguments, result)
                if result.error:
                    return
                if "error" in value:
                    # The rest of the plan, and a proposal, would rest on evidence that is missing.
                    failed = (f"planned call {name} failed: {value['error']}; plan again using the evidence, "
                              "without repeating calls whose results are already in it")
                    break
            if failed:
                previous_error = failed
                self.ctx.log("plan_rejected", reason=previous_error, replan=attempt + 1)
                continue
            # A malformed proposal is asked for again (retries.invalid_json) without rerunning the plan.
            try:
                draft = chat_json(self.ctx, "specialist", [
                    {"role": "system", "content": self.system_prompt() +
                     " The planned calls have run. Write your proposal as one JSON object whose top-level "
                     "fields are exactly the route tool's arguments (no analysis or other fields): "
                     + json.dumps(self.toolbox.argument_models["route"].model_json_schema())},
                    {"role": "user", "content": json.dumps({
                        **brief, "evidence": result.evidence, "previous_error": previous_error}, default=str)},
                ], ProposalDraft)
            except ValueError as exc:
                previous_error = str(exc)
                self.ctx.log("plan_rejected", reason=previous_error, replan=attempt + 1)
                continue
            value = self._execute(ticket_id, self.ctx.next_call_id(), "route",
                                  draft.model_dump(mode="json", exclude_defaults=True), result)
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
                # The proposal is written after the plan runs, so a planned route call is dropped rather
                # than costing a replan.
                self.ctx.log("plan_route_dropped", agent=self.name)
                continue
            calls.append((call["name"], self.toolbox.validate(call["name"], call["arguments"])))
        return calls
