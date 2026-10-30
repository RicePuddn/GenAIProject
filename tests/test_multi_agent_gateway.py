"""Specialist toolboxes, the route tool, approval tokens and mock writes (multi-agent arm)."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from pydantic import ValidationError

from support_ops.agents.models import ApprovalResult, HandoffPacket, Proposal
from support_ops.agents.toolbox import ORDERS_SHIPPING_TOOLS, RETURNS_REFUNDS_TOOLS, ExtraTool, Toolbox
from support_ops.backend.postgres import PostgresReader
from support_ops.gateway.approvals import TicketFlow
from support_ops.gateway.route import RouteTool
from support_ops.gateway.ticket_state import TicketState
from support_ops.gateway.writes import MockWrites
from support_ops.helpers.clock import Clock
from support_ops.policy.rag import RagPolicy
from support_ops.policy.rag.policy import PolicySearchArguments
from support_ops.policy.routing import RoutingRules
from support_ops.systems.tools import SupportTools
from tests.test_routing import REFUND

NOW = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
ORDER = "ORD-2026-00001"


def reader(total=80, refunds=(), delivered_days_ago=5):
    mock = Mock(spec=PostgresReader)
    mock.read.side_effect = lambda kind, record_id: (
        {"id": record_id, "total": total, "currency": "SGD", "status": "processing"} if record_id == ORDER else {"not_found": True, "kind": kind, "id": record_id})
    mock.returns_for_order.return_value = {"returns": [], "refunds": list(refunds)}
    delivered = None if delivered_days_ago is None else NOW - timedelta(days=delivered_days_ago)
    mock.shipments_for_order.return_value = [{"id": "SHP-1", "status": "delivered" if delivered_days_ago is not None else "in_transit", "delivered_at": delivered}]
    mock.items_for_order.return_value = [
        {"id": "ITM-1", "name": "Headphones", "quantity": 1, "unit_price": "189.00"},
        {"id": "ITM-2", "name": "Bottle", "quantity": 2, "unit_price": "32.00"},
    ]
    mock.payments_for_order.return_value = [{"id": "PAY-1", "amount": total, "status": "captured"}]
    for item in mock.items_for_order.return_value:
        item.update(final_sale=False, paid_amount=str(float(item["unit_price"])*item["quantity"]),
                    product_type="physical", item_status="fulfilled")
    approved = {"id": "RET-APPROVED", "status": "approved", "reason": "faulty",
                "requested_at": NOW - timedelta(days=4), "received_at": NOW - timedelta(days=3),
                "approved_at": NOW - timedelta(days=2), "reviewed_at": NOW - timedelta(days=2),
                "evidence_status": "verified", "inspection_status": "passed", "dispute_status": "none",
                "items": [{"order_item_id": item["id"], "quantity": item["quantity"], "unused": True,
                           "original_packaging": True, "tags_complete": True, "accessories_complete": True,
                           "approved_refund_amount": min(total, float(item["paid_amount"]))}
                          for item in mock.items_for_order.return_value]}
    if delivered_days_ago is not None:
        mock.returns_for_order.return_value["returns"] = [approved]
    return mock


def draft(**overrides):
    return {"action": "refund", "order_id": ORDER, "amount": 20, "items": [{"item_id": "ITM-1", "quantity": 1}],
            "cited_clauses": ["REF-1"], "route": "auto", "reason": "Damaged item.", **overrides}


class ToolboxTests(unittest.TestCase):
    def setUp(self):
        self.backend = reader()
        self.shared = SupportTools(self.backend, RagPolicy(Mock()), TicketState(Clock(NOW)))

    def test_each_specialist_sees_only_its_tools_and_never_ticket_state(self):
        for names in (ORDERS_SHIPPING_TOOLS, RETURNS_REFUNDS_TOOLS):
            box = Toolbox(self.shared, names)
            self.assertEqual([schema["function"]["name"] for schema in box.schemas], list(names))
            self.assertNotIn("ticket_state", box.names)
        box = Toolbox(self.shared, ORDERS_SHIPPING_TOOLS)
        with self.assertRaisesRegex(ValueError, "available tools: read_record"):
            box.execute("T1", "get_payments_for_order", {"order_id": ORDER})
        with self.assertRaisesRegex(ValueError, "unknown tool"):
            box.execute("T1", "ticket_state", {"action": "write", "operation": "refund", "record_id": ORDER})
        self.backend.payments_for_order.assert_not_called()

    def test_shared_tools_dispatch_and_extra_tools_get_validated_arguments(self):
        calls = []
        route = ExtraTool(Proposal, "test", lambda ticket, args: calls.append((ticket, args)) or {"ok": True})
        box = Toolbox(self.shared, RETURNS_REFUNDS_TOOLS, {"route": route})
        box.execute("T1", "get_returns_for_order", {"order_id": ORDER})
        self.backend.returns_for_order.assert_called_once_with(order_id=ORDER)
        self.assertEqual(box.schemas[-1]["function"]["name"], "route")
        with self.assertRaises(ValidationError):
            box.execute("T1", "route", {"action": "refund"})
        self.assertEqual(calls, [])

    def test_bad_subsets_are_rejected(self):
        with self.assertRaises(ValueError):
            Toolbox(self.shared, ["teleport"])
        with self.assertRaises(ValueError):
            Toolbox(self.shared, ["policy"], {"policy": ExtraTool(Proposal, "x", lambda *_: {})})


def clause(clause_id, section, via="search", references=()):
    return {"id": clause_id, "title": clause_id, "section": section, "source": "x.md", "text": "...",
            "score": None, "via": via, "references": list(references)}


class PolicySearchTests(unittest.TestCase):
    RESULT = {"query": "late refund", "clauses": [
        clause("SHP-1", "Orders and shipping", references=["SHP-1-EX-1"]),
        clause("REF-1", "Returns and refunds", references=["REF-1-EX-1"]),
        clause("ORD-1", "Orders and shipping"),
        clause("REF-1-EX-1", "Orders and shipping", via="reference"),   # exception in another section
        clause("SHP-1-EX-1", "Returns and refunds", via="reference"),
    ]}

    def search(self, sections, **args):
        retriever = Mock(search=Mock(return_value=self.RESULT["clauses"]))
        tool = RagPolicy(retriever).scoped_tool(sections)
        result = tool.run("T1", PolicySearchArguments.model_validate({"query": "late refund", **args}).model_dump())
        retriever.search.assert_called_once_with("late refund")
        return result

    def test_own_scope_keeps_its_sections_and_direct_references(self):
        result = self.search(["Returns and refunds"])
        self.assertEqual([c["id"] for c in result["clauses"]], ["REF-1", "REF-1-EX-1"])
        self.assertEqual((result["scope"], result["filtered_out"]), ("own", 3))

    def test_all_scope_and_unscoped_specialists_see_everything(self):
        self.assertEqual(len(self.search(["Returns and refunds"], scope="all")["clauses"]), 5)
        self.assertEqual(len(self.search(None)["clauses"]), 5)

    def test_no_match_in_own_sections_says_how_to_widen(self):
        result = self.search(["Account and privacy"])
        self.assertEqual(result["clauses"], [])
        self.assertIn("scope all", result["note"])

    def test_pdf_domain_names_match_scopes_and_keep_only_their_direct_references(self):
        hits = [
            clause("RET-SupportingEvidence-3.2", "returns-and-refunds",
                   references=["RET-SupportingEvidence-3.1"]),
            clause("SHP-DeliveryProblems-7.4", "orders-and-shipping",
                   references=["SHP-DeliveryProblems-7.1"]),
            clause("RET-SupportingEvidence-3.1", "returns-and-refunds", via="reference"),
            clause("SHP-DeliveryProblems-7.1", "orders-and-shipping", via="reference"),
        ]
        retriever = Mock(search=Mock(return_value=hits))
        retriever.sections.return_value = ["Orders and Shipping Policy", "Returns and Refunds Policy"]
        policy = RagPolicy(retriever)
        self.assertEqual(policy.check_scope(["Returns and refunds"]), [])
        tool = policy.scoped_tool(["Returns and refunds"])
        result = tool.run("T1", {"action": "search", "query": "unsafe evidence"})
        self.assertEqual([hit["id"] for hit in result["clauses"]],
                         ["RET-SupportingEvidence-3.2", "RET-SupportingEvidence-3.1"])


class RouteToolTests(unittest.TestCase):
    def tool(self, **backend):
        return RouteTool(reader(**backend), RoutingRules.load(), Clock(NOW), "returns_refunds")

    def test_accepted_route_returns_the_attributed_proposal_and_minimum(self):
        result = self.tool().run("T1", draft())
        self.assertEqual((result["status"], result["route"]), ("recorded", "auto"))
        self.assertEqual(result["proposal"]["specialist"], "returns_refunds")
        self.assertEqual(result["min_route"]["route"], "auto")

    def test_looser_route_is_rejected_with_the_minimum_and_stricter_is_accepted(self):
        with self.assertRaisesRegex(ValueError, "minimum is 'approver'"):
            self.tool().run("T1", draft(amount=50))
        self.assertEqual(self.tool().run("T1", draft(amount=50, route="approver"))["route"], "approver")
        self.assertEqual(self.tool().run("T1", draft(route="human"))["route"], "human")

    def test_facts_come_from_the_backend(self):
        tool = self.tool(total=80, refunds=[{"amount": "70", "status": "processed"},
                                            {"amount": "50", "status": "failed"}])
        with self.assertRaisesRegex(ValueError, "already has returns or refunds on record"):
            tool.run("T1", draft(amount=20, route="approver"))   # first shown the earlier refund
        with self.assertRaisesRegex(ValueError, "exceeds the remaining captured balance|exceeds.*balance|proposed action is blocked"):
            tool.run("T1", draft(amount=20, route="approver"))
        self.assertFalse(tool.check(Proposal(specialist="returns_refunds", **draft(amount=20)))["allowed"])
        with self.assertRaisesRegex(ValueError, "outside the ordinary return window"):
            late = self.tool(delivered_days_ago=45)
            late.reader.returns_for_order.return_value["returns"][0]["status"] = "requested"
            late.returns_checked.add(ORDER)
            late.run("T1", draft(action="return", amount=None, items=[{"item_id": "ITM-2"}]))
        with self.assertRaisesRegex(ValueError, "not found"):
            self.tool().run("T1", draft(order_id="ORD-NOPE"))
        # Information only (e.g. asking the customer to confirm the number) is fine without the order.
        self.assertEqual(self.tool().run("T1", {"action": "none", "order_id": "ORD-NOPE", "route": "auto",
                                                "reason": "Not found.", "customer_question": "Which order?"})
                         ["status"], "recorded")

    def test_returns_are_priced_from_the_order_lines_not_the_order_total(self):
        tool = self.tool(total=275.77)
        tool.reader.returns_for_order.return_value["returns"][0]["status"] = "requested"
        def minimum(*items):
            return tool.check(Proposal(specialist="returns_refunds", **draft(
                action="return", amount=None, items=[{"item_id": i, "quantity": q} for i, q in items])))
        self.assertEqual(minimum(("ITM-2", 1))["route"], "auto")   # one 32.00 bottle
        self.assertTrue(minimum(("ITM-2", 1))["allowed"])
        self.assertEqual(minimum(("ITM-1", 1))["route"], "auto")      # 189.00 headphones
        with self.assertRaisesRegex(ValueError, "not in order.*ITM-1"):
            minimum(("ITM-9", 1))
        with self.assertRaisesRegex(ValueError, "quantity 2, not 3"):
            minimum(("ITM-2", 3))

    def test_an_undelivered_shipment_past_its_estimate_raises_the_route_to_human(self):
        tool = self.tool()
        shipments = {
            "overdue": [{"id": "S", "status": "in_transit", "estimated_delivery": (NOW - timedelta(days=4)).date(),
                         "delivered_at": None}],
            "on time": [{"id": "S", "status": "in_transit", "estimated_delivery": "2026-10-05", "delivered_at": None}],
            "delivered late": [{"id": "S", "status": "delivered", "estimated_delivery": "2026-09-01",
                                "delivered_at": NOW - timedelta(days=3)}],
            "cancelled": [{"id": "S", "status": "cancelled", "estimated_delivery": "2026-09-01", "delivered_at": None}],
        }
        question = {"action": "none", "order_id": ORDER, "route": "auto", "reason": "Status shared."}
        for name, rows in shipments.items():
            with self.subTest(name):
                tool.reader.shipments_for_order.return_value = rows
                minimum = tool.check(Proposal(specialist="orders_shipping", **question))
                self.assertEqual(minimum["overdue_shipment"], name == "overdue")
                self.assertEqual(minimum["route"], "human" if name == "overdue" else "auto")

    def test_each_specialist_may_only_propose_its_own_actions(self):
        orders = RouteTool(reader(), RoutingRules.load(), Clock(NOW), "orders_shipping")
        with self.assertRaisesRegex(ValueError, "outside the orders_shipping specialist's domain.*none, order_change"):
            orders.run("T1", draft())
        orders.reader.shipments_for_order.return_value = []
        for item in orders.reader.items_for_order.return_value:
            item["item_status"] = "open"
        self.assertEqual(orders.run("T1", {"action": "order_change", "order_id": ORDER, "cited_clauses": ["ORD-1"],
                                           "route": "approver", "reason_code": "cancel", "reason": "Cancel."})["route"], "approver")
        with self.assertRaisesRegex(ValueError, "outside the returns_refunds specialist's domain"):
            self.tool().run("T1", {"action": "order_change", "order_id": ORDER, "cited_clauses": ["ORD-1"],
                                   "route": "approver", "reason_code": "cancel", "reason": "Cancel."})
        for tool in (orders, self.tool()):
            self.assertEqual(tool.run("T1", {"action": "none", "route": "human", "reason": "Not mine."})["route"],
                             "human")

    def test_existing_returns_are_shown_once_unless_the_specialist_already_looked(self):
        pending = {"returns": [{"id": "RET-1", "status": "in_transit", "reason": "unsuitable fit",
                                "requested_at": NOW - timedelta(days=20), "received_at": None}],
                   "refunds": [{"id": "RFD-9", "status": "failed", "amount": "10"}]}   # void: not listed
        question = {"action": "none", "order_id": ORDER, "route": "auto", "reason": "Asked about a return."}
        tool = self.tool()
        tool.reader.returns_for_order.return_value = pending
        with self.assertRaisesRegex(ValueError, "already has returns or refunds.*RET-1.*in_transit"):
            tool.run("T1", question)
        result = tool.run("T1", question)
        self.assertEqual([r["id"] for r in result["min_route"]["existing_requests"]], ["RET-1"])

        looked = RouteTool(tool.reader, RoutingRules.load(), Clock(NOW), "returns_refunds", returns_checked={ORDER})
        self.assertEqual(looked.run("T1", question)["status"], "recorded")   # no bounce: it already looked

    def test_a_refund_is_capped_at_what_was_paid_for_its_items(self):
        with self.assertRaisesRegex(ValueError, "refund 40.00 exceeds the 32.00 paid for the listed items"):
            self.tool().run("T1", draft(amount=40, route="approver", items=[{"item_id": "ITM-2", "quantity": 1}]))
        both = self.tool().run("T1", draft(amount=64, route="approver", items=[{"item_id": "ITM-2", "quantity": 2}]))
        self.assertEqual(both["status"], "recorded")
        no_items = self.tool().check(Proposal(specialist="returns_refunds", **draft(items=[])))
        self.assertEqual(no_items["route"], "auto")

    def test_a_refund_on_a_shipped_parcel_that_has_not_arrived(self):
        tool = self.tool()
        late = {"id": "S", "status": "in_transit", "shipped_at": NOW - timedelta(days=9),
                "estimated_delivery": (NOW - timedelta(days=2)).date(), "delivered_at": None}
        for name, shipment, expected in (("in transit", {**late, "estimated_delivery": (NOW + timedelta(days=2)).date()},
                                          "human"), ("overdue", late, "human")):
            with self.subTest(name):
                tool.reader.shipments_for_order.return_value = [shipment]
                self.assertEqual(tool.check(Proposal(specialist="returns_refunds", **draft()))["route"], expected)

    def test_information_only_proposal_reads_nothing(self):
        tool = self.tool()
        tool.run("T1", {"action": "none", "route": "auto", "reason": "Tracking shared."})
        tool.reader.read.assert_not_called()


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock(NOW)
        self.flow = TicketFlow(TicketState(self.clock), self.clock)
        self.writes = MockWrites(self.flow)
        self.rules = RoutingRules.load()

    def propose(self, ticket="T1", **overrides):
        proposal = Proposal(specialist="returns_refunds", **draft(**overrides))
        self.flow.transition(ticket, "triaged")
        self.flow.transition(ticket, "proposed")
        return proposal, self.rules.min_route(proposal.action, proposal.amount, goods_value=1000, business_facts=REFUND)

    def write(self, token, ticket="T1", amount=20, key="K1", order=ORDER):
        return self.writes.execute(ticket, token=token, operation="refund", order_id=order,
                                   amount=amount, idempotency_key=key)

    def test_auto_refund_executes_once_and_replays_on_retry(self):
        token = self.flow.approve("T1", *self.propose())["approval_token"]
        first = self.write(token)
        self.assertEqual((first["status"], self.flow.status("T1")), ("executed", "executed"))
        again = self.write(token)
        self.assertTrue(again["replayed"])
        self.assertEqual(again["write_id"], first["write_id"])
        self.assertEqual(len(self.writes.executed), 1)
        self.flow.transition("T1", "resolved")

    def test_token_bound_to_ticket_order_and_amount(self):
        token = self.flow.approve("T1", *self.propose())["approval_token"]
        for name, kwargs in {"amount": {"amount": 25}, "order": {"order": "ORD-2"}, "ticket": {"ticket": "T2"}}.items():
            with self.subTest(name):
                self.assertEqual(self.write(token, key=name, **kwargs)["status"], "refused")
        self.assertEqual(self.write(token)["status"], "executed")
        self.assertEqual(self.write("forged", key="K2")["status"], "refused")

    def test_expired_and_used_tokens_are_refused(self):
        token = self.flow.approve("T1", *self.propose())["approval_token"]
        self.clock.advance(301)
        self.assertIn("expired", self.write(token)["reason"])
        token = self.flow.approve("T2", *self.propose(ticket="T2"))["approval_token"]
        self.flow.redeem(token, "T2", "refund", ORDER, 20)
        self.assertIn("already used", self.write(token, ticket="T2", key="K2")["reason"])

    def test_approver_route_needs_a_pre_execution_approval(self):
        proposal, minimum = self.propose(amount=50, route="approver")
        with self.assertRaisesRegex(ValueError, "pre-execution approval"):
            self.flow.approve("T1", proposal, minimum)
        reply_check = ApprovalResult(stage="reply", decision="approve", reason="ok")
        with self.assertRaises(ValueError):
            self.flow.approve("T1", proposal, minimum, reply_check)
        ok = ApprovalResult(stage="pre_execution", decision="approve", reason="Within policy.")
        self.assertIsNotNone(self.flow.approve("T1", proposal, minimum, ok)["approval_token"])

    def test_looser_or_human_routes_get_no_token(self):
        proposal, minimum = self.propose(amount=50)   # auto chosen, approver required
        with self.assertRaisesRegex(ValueError, "looser"):
            self.flow.approve("T1", proposal, minimum)
        proposal, minimum = self.propose(ticket="T2", amount=150, route="human")
        with self.assertRaisesRegex(ValueError, "handed off|not allowed"):
            self.flow.approve("T2", proposal, minimum)
        self.assertEqual(self.flow.tokens, {})

    def test_transitions_are_enforced_and_every_refusal_is_audited(self):
        with self.assertRaises(ValueError):
            self.flow.transition("T1", "executed")
        token = self.flow.approve("T1", *self.propose())["approval_token"]
        self.write(token)
        self.write("forged", key="K2")
        refused = [event for event in self.flow.audit_log if event.get("outcome") == "refused"]
        self.assertEqual([event["type"] for event in refused], ["transition", "write"])
        self.assertTrue(all(token not in str(event) for event in self.flow.audit_log))

    def test_write_outside_the_approved_state_is_refused_without_using_the_token(self):
        token = self.flow.approve("T1", *self.propose())["approval_token"]
        self.flow.transition("T1", "awaiting_human")
        self.assertIn("state awaiting_human", self.write(token)["reason"])
        self.assertFalse(self.flow.tokens[token]["used"])

    def test_handoff_records_the_structured_packet(self):
        proposal, _ = self.propose(amount=150, route="human")
        result = self.flow.handoff(HandoffPacket(
            ticket_id="T1", reason="route: human", summary="Refund of 150 SGD requested.",
            recommendation="Check photos.", proposal=proposal))
        self.assertEqual((result["status"], self.flow.status("T1")), ("awaiting_human", "awaiting_human"))
        self.assertEqual(result["handoff_packet"]["proposal"]["amount"], 150)
        with self.assertRaises(ValueError):
            self.flow.transition("T1", "approved")


if __name__ == "__main__":
    unittest.main()
