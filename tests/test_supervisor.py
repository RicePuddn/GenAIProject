"""The supervisor's deterministic parts: handoff triggers, the handoff packet and the reply outcome."""
import unittest

from support_ops.agents.models import Proposal, Triage
from support_ops.supervisor.escalation import handoff_packet, triage_escalation
from support_ops.supervisor.outcome import customer_outcome, verified_facts

TRIAGE = {"intent": "refund_request", "urgency": "normal", "language": "en",
          "specialist": "returns_refunds", "confidence": 0.9, "reason": "Refund."}
RETURN = Proposal(specialist="returns_refunds", action="return", order_id="ORD-1", cited_clauses=["RET-1"],
                  items=[{"item_id": "ITM-1-2", "quantity": 1}], route="approver", reason="Dented on arrival.")
ITEMS = {"agent": "returns_refunds", "tool": "get_items_for_order",
         "result": {"items": [{"id": "ITM-1-2", "name": "Terra Insulated Bottle", "quantity": 2}]}}
ROUTE = {"agent": "returns_refunds", "tool": "route", "result": {"status": "recorded"}}
FAILED = {"agent": "returns_refunds", "tool": "get_payments_for_order", "result": {"error": "timeout"}}


def outcome(**overrides):
    args = {"status": "executed", "evidence": [ITEMS, ROUTE], "proposal": RETURN,
            "execution": {"status": "executed"}, "min_route": None, "customer_question": None,
            "currency": "SGD", **overrides}
    return customer_outcome(**args)


class EscalationTests(unittest.TestCase):
    def test_triage_triggers_in_order_and_none_when_clear(self):
        self.assertIsNone(triage_escalation(Triage(**TRIAGE), 0.5))
        cases = {"customer asked for a human": {"wants_human": True, "duplicate_of": "T0"},
                 "duplicate of open ticket T0": {"duplicate_of": "T0"},
                 "no specialist covers this request": {"specialist": None},
                 "low triage confidence (0.30)": {"confidence": 0.3}}
        for reason, overrides in cases.items():
            with self.subTest(reason):
                self.assertEqual(triage_escalation(Triage(**{**TRIAGE, **overrides}), 0.5), reason)

    def test_packet_lists_actions_citations_and_the_recommendation(self):
        packet = handoff_packet(ticket_id="T1", reason="route: human", customer_message="Bottle dented",
                                triage=Triage(**TRIAGE), proposal=RETURN, min_route={"route": "approver"},
                                evidence=[ITEMS, FAILED], execution={"status": "refused", "operation": "return"})
        self.assertEqual(packet.actions_taken, ["returns_refunds: get_items_for_order",
                                                "returns_refunds: get_payments_for_order (error)",
                                                "write refused: return"])
        self.assertEqual((packet.cited_clauses, packet.summary), (["RET-1"], "refund_request: Bottle dented"))
        self.assertIn("return via approver", packet.recommendation)

    def test_packet_before_any_proposal(self):
        packet = handoff_packet(ticket_id="T1", reason="guard: malicious", customer_message="x", triage=None,
                                proposal=None, min_route=None, evidence=[], execution=None)
        self.assertEqual(packet.summary, "request: x")
        self.assertIn("stopped before a proposal", packet.recommendation)


class OutcomeTests(unittest.TestCase):
    def test_facts_leave_out_the_route_and_failed_calls(self):
        self.assertEqual([fact["tool"] for fact in verified_facts([ITEMS, ROUTE, FAILED])], ["get_items_for_order"])

    def test_an_executed_return_is_told_in_product_terms(self):
        result = outcome()
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["request"], {"action": "return", "order_id": "ORD-1",
                                             "items": [{"product": "Terra Insulated Bottle", "quantity": 1}]})
        self.assertIn("recorded and complete", result["tell_customer"])

    def test_what_to_tell_the_customer_follows_the_status(self):
        cases = {
            "share what the facts show": dict(status="awaiting_human", execution=None),
            "Promise nothing else": dict(status="awaiting_human", execution=None, evidence=[ROUTE]),
            "Ask: Which bottle?": dict(status="awaiting_customer", execution=None, customer_question="Which bottle?"),
            "if none are listed, do not apologise": dict(status="approved", execution=None),
        }
        for phrase, overrides in cases.items():
            with self.subTest(phrase):
                self.assertIn(phrase.lower(), outcome(**overrides)["tell_customer"].lower())

    def test_a_late_shipment_becomes_a_problem_to_apologise_for(self):
        result = outcome(status="awaiting_human", execution=None, min_route={"overdue_shipment": True})
        self.assertEqual(result["problems"], ["The shipment is past its estimated delivery date."])
        self.assertNotIn("problems", outcome())


if __name__ == "__main__":
    unittest.main()
