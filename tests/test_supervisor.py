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
        # Off topic is not a handoff, even with no specialist; asking for a person still is.
        self.assertIsNone(triage_escalation(Triage(**{**TRIAGE, "specialist": None, "off_topic": True}), 0.5))
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


class MissingRecordTests(unittest.TestCase):
    def test_missing_records_come_from_lookups_and_route_errors_once_each(self):
        from support_ops.supervisor.escalation import missing_records

        evidence = [
            {"tool": "read_record", "result": {"not_found": True, "kind": "orders", "id": "ORD-9"}},
            {"tool": "route", "result": {"error": "order ORD-9 not found: ask the customer to confirm"}},
            {"tool": "read_record", "result": {"not_found": True, "kind": "payments", "id": "PAY-9"}},
            {"tool": "read_record", "result": {"id": "ORD-1", "total": 5}},
        ]
        self.assertEqual(missing_records(evidence), [("order", "ORD-9"), ("payment", "PAY-9")])

    def test_insisting_means_the_same_id_in_an_earlier_customer_turn(self):
        from support_ops.supervisor.escalation import customer_insists

        history = [{"role": "customer", "content": "It's ORD-9"}, {"role": "agent", "content": "ORD-9 not found"}]
        self.assertTrue(customer_insists(history, "Yes, ORD-9 is right", "ORD-9"))
        self.assertTrue(customer_insists(history, "it's ord ‑ 9", "ORD-9"))   # dash, case and spacing vary
        self.assertFalse(customer_insists(history[1:], "ORD-9", "ORD-9"))   # only the agent mentioned it
        # A different number now: a new question, even if the specialist re-checked the old ID.
        self.assertFalse(customer_insists(history, "Sorry, it's ORD-90", "ORD-9"))

    def test_the_question_names_the_id_the_customer_just_gave(self):
        from support_ops.supervisor.escalation import confirm_question

        missing = [("order", "ORD-000042"), ("order", "ORD-00042")]
        self.assertIn("ORD-00042.", confirm_question(missing, "ORD‑00042"))
        self.assertIn("ORD-000042.", confirm_question(missing))


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
            "invite a question about those": dict(status="awaiting_customer", execution=None, proposal=None,
                                                  evidence=[], off_topic=True),
            "if none are listed, do not apologise": dict(status="approved", execution=None),
        }
        for phrase, overrides in cases.items():
            with self.subTest(phrase):
                self.assertIn(phrase.lower(), outcome(**overrides)["tell_customer"].lower())

    def test_a_late_shipment_becomes_a_problem_to_apologise_for(self):
        result = outcome(status="awaiting_human", execution=None, min_route={"overdue_shipment": True})
        self.assertEqual(result["problems"], ["The shipment is past its estimated delivery date."])
        self.assertNotIn("problems", outcome())

    def test_a_confirmed_missing_record_is_not_asked_about_again(self):
        result = outcome(status="awaiting_human", execution=None, proposal=None, evidence=[],
                         unfound_record="order ORD-9")
        self.assertIn("confirmed order ORD-9", result["problems"][0])
        self.assertIn("do not ask them to check it again", result["problems"][0].lower())
        self.assertIn("acknowledge every item in problems", result["tell_customer"].lower())


if __name__ == "__main__":
    unittest.main()
