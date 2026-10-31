"""Typed messages between the multi-agent roles (agents/models.py)."""
import json
import unittest

from pydantic import ValidationError

from support_ops.agents.models import ApprovalResult, FirstDecision, HandoffPacket, Proposal, SupervisorDecision

REFUND = {
    "specialist": "returns_refunds", "action": "refund", "order_id": "ORD-2026-00001",
    "amount": 25, "reason_code": "damaged_item", "cited_clauses": ["REF-1"],
    "route": "auto", "reason": "Damaged on arrival, within the window.",
}


class ProposalTests(unittest.TestCase):
    def test_refund_proposal_parses_from_model_json(self):
        proposal = Proposal.model_validate_json(
            '{"specialist":"returns_refunds","action":"refund","order_id":"ORD-2026-00001",'
            '"amount":"25.50","cited_clauses":["REF-1"],"route":"approver","reason":"late"}')
        self.assertEqual(proposal.amount, 25.5)
        self.assertFalse(proposal.policy_exception)

    def test_write_actions_need_their_facts(self):
        broken = {
            "refund without amount": {**REFUND, "amount": None},
            "write without order": {**REFUND, "order_id": None},
            "write without citation": {**REFUND, "cited_clauses": []},
            "return without items": {**REFUND, "action": "return", "amount": None},
            "item listed twice": {**REFUND, "action": "return", "amount": None,
                                  "items": [{"item_id": "ITM-1"}, {"item_id": "ITM-1"}]},
            "zero quantity": {**REFUND, "action": "return", "amount": None,
                              "items": [{"item_id": "ITM-1", "quantity": 0}]},
            "zero amount": {**REFUND, "amount": 0},
            "unknown route": {**REFUND, "route": "whenever"},
            "unknown field": {**REFUND, "approved": True},
        }
        for name, data in broken.items():
            with self.subTest(name), self.assertRaises(ValidationError):
                Proposal.model_validate(data)

    def test_fields_an_action_does_not_use_are_dropped_and_item_aliases_accepted(self):
        ret = Proposal.model_validate({**REFUND, "action": "return", "items": [{"order_item_id": "ITM-1"}]})
        self.assertEqual((ret.amount, ret.items[0].item_id), (None, "ITM-1"))
        refund = Proposal.model_validate({**REFUND, "items": [{"id": "ITM-1"}]})   # refunds keep their items
        self.assertEqual(refund.items[0].item_id, "ITM-1")
        self.assertEqual(Proposal.model_validate({**REFUND, "action": "order_change", "amount": None,
                                                  "items": [{"id": "ITM-1"}]}).items, [])
        question = Proposal(specialist="returns_refunds", action="none", route="auto", reason="Need detail.",
                            items=[{"item_id": "ITM-1"}], customer_question="Is it unused?")
        self.assertEqual(question.items, [])
        schema = Proposal.model_json_schema()
        self.assertIn("ITM-0031-2", json.dumps(schema))
        self.assertIn("item_id", json.dumps(schema))

    def test_information_only_proposal_needs_no_order_or_citation(self):
        proposal = Proposal(specialist="orders_shipping", action="none", route="auto",
                            reason="Tracking shared.")
        self.assertIsNone(proposal.order_id)


class OtherModelTests(unittest.TestCase):
    def test_a_decision_matches_its_action(self):
        task = {"specialist": "orders_shipping", "request": "Where is ORD-1?"}
        base = dict(thought="t", reason="r")
        self.assertEqual(SupervisorDecision(**base, action="delegate", tasks=[task]).tasks[0].request, "Where is ORD-1?")
        self.assertEqual(SupervisorDecision(**base, action="finish", tasks=[task]).tasks, [])
        for bad in ([], [task, task]):
            with self.subTest(bad), self.assertRaises(ValidationError):
                SupervisorDecision(**base, action="delegate", tasks=bad)
        with self.assertRaises(ValidationError):
            SupervisorDecision(**base, action="delegate", tasks=[{"specialist": "orders_shipping"}])

    def test_the_first_decision_must_state_the_case(self):
        base = dict(thought="t", reason="r", action="redirect")
        with self.assertRaises(ValidationError):
            FirstDecision(**base)
        case = {"issue": "Cancel.", "intent": "cancel", "urgency": "low", "language": "en", "order_ids": ["ORD-1"]}
        self.assertEqual(FirstDecision(**base, case=case).case.order_ids, ["ORD-1"])

    def test_rejection_or_escalation_must_name_an_issue(self):
        ApprovalResult(stage="reply", decision="approve", reason="Matches the executed refund.")
        for decision in ("reject", "escalate"):
            with self.subTest(decision), self.assertRaises(ValidationError):
                ApprovalResult(stage="pre_execution", decision=decision, reason="No.")

    def test_handoff_packet_carries_the_proposal(self):
        packet = HandoffPacket(ticket_id="T1", reason="route: human", summary="Refund of 150 SGD.",
                               recommendation="Approve if photos confirm damage.",
                               proposal=Proposal.model_validate({**REFUND, "amount": 150, "route": "human"}))
        self.assertEqual(packet.model_dump()["proposal"]["amount"], 150)


if __name__ == "__main__":
    unittest.main()
