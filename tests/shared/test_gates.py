"""The shared gates: guard, approver, reply wording and reply check, ticket save, handoff packet, prompts."""
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from support_ops import prompts
from support_ops.shared.flow import Proposal
from support_ops.shared.gates.approver import Approver
from support_ops.shared.gates.guard import screen
from support_ops.shared.gates.handoff import handoff_packet
from support_ops.shared.gates.reply import (internal_id_issue, problems, refund_wording_issue, wording_issues,
                                            write_reply)
from support_ops.shared.gates.ticket import category, confirmed_order, save_ticket
from tests.fakes import NOW, ORDER, FakeDatabase, ScriptedLLM, as_json, reject, services, text

RECORDED = [{"tool": "backend_write", "result": {"operation": "refund", "status": "executed"}}]


# ----------------------------------------------------------------------------- guard


def test_the_guard_screens_only_the_current_message_once():
    llm = ScriptedLLM().on("guard", lambda *_: text("0.99"))
    ctx = services(llm=llm).ctx
    assert screen(ctx, "hello") == "malicious"
    guard_calls = [(messages, tools) for role, messages, tools in llm.calls if role == "guard"]
    assert guard_calls == [([{"role": "user", "content": "hello"}], None)]


def test_the_guard_fails_closed_and_can_be_switched_off():
    for label in ("", "maybe", "1.5"):
        ctx = services(llm=ScriptedLLM().on("guard", lambda *_, label=label: text(label))).ctx
        assert screen(ctx, "hi") == "error"
    ctx = services(llm=ScriptedLLM().on("guard", lambda *_: text("benign"))).ctx
    assert screen(ctx, "hi") == "benign"
    tools = services(llm=ScriptedLLM().on("guard", lambda *_: text("0.99")))
    tools.ctx.cfg["guard"]["enabled"] = False
    assert screen(tools.ctx, "hi") == "benign"


# ----------------------------------------------------------------------------- approver


def test_unreadable_approver_output_counts_as_escalate():
    ctx = services(llm=ScriptedLLM().push("approver", text("approve!"), text("{}"))).ctx
    verdict = Approver(ctx).check_reply({"message": "hi"}, {"status": "resolved"}, "Hello.")
    assert verdict.decision == "escalate" and "unreadable approver output" in verdict.issues[0]
    assert len(ctx.tracer.of_kind("json_retry")) == 1   # one retry, then it is read as an escalation


def test_both_checks_use_their_one_prompt_and_set_the_stage_in_code():
    llm = ScriptedLLM().push("approver", as_json({"decision": "approve", "issues": [], "reason": "ok", "stage": "x"}))
    ctx = services(llm=llm).ctx
    Approver(ctx).check_write({"message": "refund"}, {"write": {}, "rule_check": {}, "evidence": []})
    assert llm.calls[0][1][0]["content"] == prompts.load("approver", "write")
    assert ctx.tracer.of_kind("approval")[0]["stage"] == "write"


# ----------------------------------------------------------------------------- reply


def test_a_pending_refund_described_as_paid_is_caught():
    for reply in ("The payment of SGD 32.00 has been refunded to the original Visa card.",
                  "Your refund has been processed.", "The money has been returned to your card.",
                  "Your refund was issued today."):
        assert "only recorded as pending" in refund_wording_issue(reply, RECORDED)
    for reply in ("A refund of SGD 32.00 is recorded and pending.",
                  "Your refund will go back to the original card; the policy estimates 5 to 10 business days."):
        assert refund_wording_issue(reply, RECORDED) is None
    assert refund_wording_issue("Your refund has been processed.", []) is None   # no refund here
    processed = [{"tool": "get_returns_for_order", "result": {"returns": [], "refunds": [{"status": "processed"}]}}]
    assert refund_wording_issue("Your refund has been processed.", processed) is None   # true
    pending = [{"tool": "get_returns_for_order", "result": {"returns": [], "refunds": [{"status": "pending"}]}}]
    assert refund_wording_issue("Your refund has been processed.", pending) is not None


def test_internal_ids_are_caught_but_the_order_number_may_stay():
    for reply in ("according to our policy (section 4.2, ODR‑OrderChangeCancel‑4.2), it should appear",
                  "Under REF-Refunds-6.6 you qualify.", "Refund RFD-W-EC7B6EA3E4FA is pending.",
                  "We checked item ITM-EVAL-101-1 and payment PAY-EVAL-109.", "Your ticket is TKT-E139AC2B96F0."):
        assert "internal IDs" in internal_id_issue(reply)
    for reply in ("Your order ORD‑2026‑00109 has been cancelled.", "Your Terra Insulated Bottle (SKU AUR-A100) is eligible.",
                  "Under our cancellation policy the refund is pending."):
        assert internal_id_issue(reply) is None
    assert len(wording_issues("Your refund has been processed under REF-Refunds-6.6.", RECORDED)) == 2


def test_a_wording_slip_is_redrafted_without_spending_a_reply_check():
    llm = ScriptedLLM().push("reply_writer", text("Your SGD 20 has been refunded to your card."),
                             text("A refund of SGD 20 is recorded and pending."))
    ctx = services(llm=llm).ctx
    reply, review = write_reply(ctx, Approver(ctx), {"message": "refund"},
                                {"status": "resolved", "note": None, "evidence": RECORDED})
    assert reply == "A refund of SGD 20 is recorded and pending." and review.decision == "approve"
    assert "only recorded as pending" in llm.user_payload("reply_writer", 1)["previous_draft_issues"][0]
    assert llm.roles().count("approver") == 1   # only the good draft was checked


def test_wording_rewrites_do_not_use_up_the_reply_check_s_drafts():
    llm = ScriptedLLM().push("reply_writer", text("Under REF-Refunds-6.6 you qualify."),
                             text("Under ODR-OrderChangeCancel-4.2 you qualify."), text("Vague."), text("Clear."))
    llm.push("approver", reject("too vague"))
    ctx = services(llm=llm).ctx
    reply, _ = write_reply(ctx, Approver(ctx), {"message": "?"}, {"status": "resolved", "note": None, "evidence": []})
    assert reply == "Clear."   # two wording rewrites, then two checked drafts
    assert len(ctx.tracer.of_kind("reply_wording_rejected")) == 2


def test_no_draft_passing_returns_none_for_the_hold_reply():
    llm = ScriptedLLM().push("approver", reject("wrong"), reject("still wrong"))
    ctx = services(llm=llm).ctx
    reply, review = write_reply(ctx, Approver(ctx), {"message": "?"}, {"status": "resolved", "note": None, "evidence": []})
    assert reply is None and review.issues == ["still wrong"]


def test_the_reply_writer_sees_the_ticket_and_the_workflow_date():
    llm = ScriptedLLM()
    ctx = services(llm=llm).ctx
    ticket = {"status": "awaiting_customer", "note": "Which bottle?", "evidence": []}
    write_reply(ctx, Approver(ctx), {"message": "refund"}, ticket)
    assert llm.user_payload("reply_writer")["ticket"] == ticket
    assert "Current workflow date: 2026-10-01." in llm.calls[0][1][0]["content"]


# ----------------------------------------------------------------------------- ticket and handoff


def test_the_ticket_category_and_order_come_from_what_the_turn_recorded():
    write = {"tool": "backend_write", "arguments": {"operation": "cancel"},
             "result": {"status": "executed", "order_id": "ORD-2"}}
    lookup = {"tool": "read_record", "arguments": {}, "result": {"id": "ORD-1"}}
    assert (category([lookup, write]), confirmed_order([lookup, write])) == ("cancellation", "ORD-2")
    assert (category([lookup]), confirmed_order([lookup])) == ("other", "ORD-1")
    shipments = {"tool": "get_shipments_for_order", "arguments": {}, "result": {"shipments": []}}
    assert category([shipments]) == "delivery"
    missing = {"tool": "read_record", "arguments": {}, "result": {"not_found": True, "id": "ORD-9"}}
    assert confirmed_order([missing]) is None   # a claimed order is never linked


def test_a_failed_ticket_save_is_logged_and_never_raised():
    class Broken(FakeDatabase):
        def save_ticket(self, **ticket):
            raise RuntimeError("database down")

    tools = services()
    save_ticket(tools.ctx, Broken(), first_message="hi", status="resolved", reply="Hello.", handoff_packet=None)
    assert tools.ctx.tracer.of_kind("ticket_save_failed")[0]["error"] == "database down"


def test_the_ticket_keeps_the_conversation_s_first_message():
    db = FakeDatabase()
    tools = services(db)
    save_ticket(tools.ctx, db, first_message="My   order\nis late", status="awaiting_human", reply="Hold.",
                handoff_packet={"reason": "x"})
    saved = db.ticket_saves[0]
    assert (saved["subject"], saved["customer_message"], saved["resolution"]) == ("My order is late",
                                                                                   "My   order\nis late", None)


def test_the_handoff_packet_lists_every_call_write_and_citation():
    evidence = [{"agent": "returns_refunds", "tool": "policy", "arguments": {}, "result": {"clauses": []}},
                {"agent": "returns_refunds", "tool": "backend_write",
                 "arguments": {"operation": "refund", "cited_clauses": ["REF-Refunds-6.1"]},
                 "result": {"status": "awaiting_human", "reason": "limit"}},
                {"agent": "returns_refunds", "tool": "get_items_for_order", "arguments": {}, "result": {"error": "x"}}]
    packet = handoff_packet(ticket_id="T1", reason="limit", message="Refund", evidence=evidence, note="Needs a person.")
    assert packet["actions_taken"] == ["returns_refunds: policy", "returns_refunds: backend_write",
                                       "returns_refunds: get_items_for_order (error)"]
    assert packet["cited_clauses"] == ["REF-Refunds-6.1"] and packet["writes"][0]["status"] == "awaiting_human"
    assert packet["recommendation"] == "Needs a person."
    json.dumps(packet, default=str)


# ----------------------------------------------------------------------------- prompts and proposals


def test_prompts_need_every_placeholder_and_use_every_value():
    with pytest.raises(ValueError, match="needs"):
        prompts.load("reply_writer")
    with pytest.raises(ValueError, match="no placeholder"):
        prompts.load("reply_writer", today="2026-10-01", tomorrow="x")
    with pytest.raises(KeyError):
        prompts.load("approver", "nonsense")
    assert '{"decision": "approve"' in prompts.load("approver", "write")   # JSON braces are not placeholders


def test_a_proposal_survives_the_usual_model_habits():
    data = {"analysis": "thinking...", "proposal": {"outcome": "resolved", "note": "Done.", "writes": None,
                                                   "reasoning": "x"}}
    proposal = Proposal.model_validate(data)
    assert (proposal.outcome, proposal.writes) == ("resolved", [])
    write = Proposal.model_validate({"outcome": "resolved", "note": "x", "writes": [
        {"operation": "refund", "order_id": ORDER, "amount": "20", "reason": "damaged",
         "items": [{"order_item_id": "ITM-1"}], "cited_clauses": ["refunds.md"]}]}).writes[0]
    assert (write.amount, write.items[0].item_id, write.items[0].quantity) == (20.0, "ITM-1", 1)
    with pytest.raises(ValueError):
        Proposal.model_validate({"outcome": "resolved", "note": "x", "route": "auto"})   # unknown fields


def test_problems_come_from_what_the_agents_looked_up():
    today = date(2026, 10, 1)
    evidence = [
        {"tool": "get_shipments_for_order", "result": {"shipments": [
            {"status": "in_transit", "estimated_delivery": date(2026, 9, 28), "delivered_at": None},
            {"status": "delivered", "estimated_delivery": date(2026, 9, 20), "delivered_at": datetime(2026, 9, 21)},
            {"status": "in_transit", "estimated_delivery": "2026-10-03", "delivered_at": None}]}},
        {"tool": "get_returns_for_order", "result": {"returns": [
            {"status": "received", "requested_at": datetime(2026, 9, 25, tzinfo=timezone.utc)},
            {"status": "approved", "requested_at": datetime(2026, 9, 20)}], "refunds": []}},
        {"tool": "get_shipments_for_order", "result": {"error": "timeout"}},
    ]
    assert problems(evidence, today) == ["A shipment is past its estimated delivery date (2026-09-28).",
                                         "A return the customer requested on 2026-09-25 has not been processed by the store yet."]
    assert problems([], today) == []   # nothing looked up, nothing listed


def test_the_reply_writer_and_the_reply_check_both_see_the_problems():
    from tests.setups.scripts import Scenario, run, script
    output, _ = run("s7-a", llm := script("s7-a", Scenario(reads=[("get_returns_for_order", {"order_id": ORDER})],
                                                        specialist="returns_refunds")))
    ticket = llm.user_payload("reply_writer")["ticket"]
    assert "problems" in ticket and llm.user_payload("approver")["ticket"]["problems"] == ticket["problems"]


# ----------------------------------------------------------------------------- first return requests


def first_return(reason, *, item="ITM-1", db=None):
    db = db if db is not None else FakeDatabase()
    db.tables["returns"].clear()
    return services(db), {"operation": "return_request", "order_id": ORDER, "reason": reason,
                          "items": [{"item_id": item, "quantity": 1}], "cited_clauses": ["refunds.md"]}


@pytest.mark.parametrize("reason", ["change_of_mind", "damaged", "faulty", "wrong_item", "missing_item"])
def test_a_first_return_request_passes_on_the_rules_alone(reason):
    service, args = first_return(reason)
    facts = service.gate.facts.read("return_request", ORDER, ["ITM-1"], stated_reason=reason)
    decision = service.gate.rules.check("return_request", facts=facts)
    assert (decision["check"], decision["route"]) == ("return_request.new_request", "auto")


def test_a_first_return_request_for_an_unknown_reason_can_be_corrected():
    service, args = first_return("damaged_item")
    with pytest.raises(ValueError, match="exactly one of"):
        service.gate.check(args)
    assert not service.ctx.stopped


def test_a_change_of_mind_on_final_sale_goods_is_refused_but_a_fault_is_not():
    db = FakeDatabase()
    db.tables["order_items"]["ITM-1"]["final_sale"] = True
    service, args = first_return("change_of_mind", db=db)
    assert service.gate.check(args)["policy_check"]["check"] == "return_request.final_sale"
    service, args = first_return("faulty", db=db)
    facts = service.gate.facts.read("return_request", ORDER, ["ITM-1"], stated_reason="faulty")
    assert service.gate.rules.check("return_request", facts=facts)["allowed"]


def test_a_first_return_request_past_the_window_or_for_a_digital_item_goes_to_a_person():
    db = FakeDatabase()
    db.tables["shipments"]["SHP-1"]["delivered_at"] = NOW - timedelta(days=31)
    service, args = first_return("faulty", db=db)
    assert service.gate.check(args)["status"] == "awaiting_human"
    db = FakeDatabase()
    db.tables["order_items"]["ITM-1"]["product_type"] = "digital"
    service, args = first_return("faulty", db=db)
    assert service.gate.check(args)["status"] == "awaiting_human"


def test_a_counterfeit_claim_on_a_first_return_request_goes_to_a_person():
    service, args = first_return("counterfeit")
    assert service.gate.check(args)["status"] == "awaiting_human"


def test_a_refund_waits_for_the_return_to_be_approved():
    db = FakeDatabase()
    db.tables["returns"]["RET-1"].update(status="requested", approved_at=None, reviewed_at=None,
                                         inspection_status="pending", evidence_status="pending")
    service = services(db)
    facts = service.gate.facts.read("refund", ORDER, ["ITM-1", "ITM-2"])
    decision = service.gate.rules.check("refund", 20, facts)
    assert (decision["check"], decision["allowed"]) == ("refund.request_not_accepted", False)
