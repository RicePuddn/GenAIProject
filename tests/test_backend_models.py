"""Scenario integrity and gold decisions, using the same rows seeded into PostgreSQL."""
import copy
import json
import os
import unittest
from datetime import datetime
from decimal import Decimal

from support_ops.backend.postgres import PostgresReader
from support_ops.helpers.clock import Clock
from support_ops.helpers.config import ROOT
from support_ops.policy.facts import BackendFacts
from support_ops.policy.rag.loader import load_corpus
from support_ops.policy.rules import Rules
from scripts.seed_backend import SEED_TABLES


# Read the shared fixture labels directly; no agent evaluation runner is needed.
def load_gold():
    return json.loads((ROOT / "data/cases/agents/gold.json").read_text())


# This read-only test adapter returns the same joins as PostgresReader, without a database.
class ScenarioReader:
    def __init__(self):
        self.tables = {table: [dict(zip(columns, values, strict=True)) for values in rows]
                       for table, (columns, rows) in SEED_TABLES.items()}

    def read(self, kind, record_id):
        return copy.deepcopy(next((row for row in self.tables[kind] if row["id"] == record_id),
                                  {"not_found": True, "id": record_id, "kind": kind}))

    def items_for_order(self, order_id):
        return [{**row, **{key: self.read("products", row["product_id"])[key]
                           for key in ("name", "sku", "category", "product_type")}}
                for row in self.tables["order_items"] if row["order_id"] == order_id]

    def shipments_for_order(self, order_id):
        return [{**row, "items": [line for line in self.tables["shipment_items"] if line["shipment_id"] == row["id"]]}
                for row in self.tables["shipments"] if row["order_id"] == order_id]

    def returns_for_order(self, order_id):
        return {"returns": [{**row, "items": [line for line in self.tables["return_items"] if line["return_id"] == row["id"]]}
                            for row in self.tables["returns"] if row["order_id"] == order_id],
                "refunds": [row for row in self.tables["refunds"] if row["order_id"] == order_id]}

    def payments_for_order(self, order_id):
        return [row for row in self.tables["payments"] if row["order_id"] == order_id]


# Validate business outcomes independently of an LLM's exact wording or tool ordering.
def check_gold(test, reader):
    gold, rules = load_gold(), Rules.load()
    source = BackendFacts(reader, Clock(datetime.fromisoformat(gold["clock_start"])))
    for case in gold["cases"]:
        with test.subTest(case=case["id"]):
            expected = case["expected"]
            if expected["operation"] == "delivery":
                checks = [rules.check("delivery", facts=parcel) for parcel in source.deliveries(case["order_id"])]
                actual = next((check for check in checks if check["requires_human"]), checks[0])
            else:
                facts = source.read(expected["operation"], case["order_id"], case["item_ids"])
                actual = rules.check(expected["operation"], expected.get("amount"), facts)
            test.assertEqual(actual["check"], expected["rule_check"])
            test.assertEqual(actual["route"], expected["minimum_route"])
            test.assertEqual(actual["allowed"], expected["allowed"])
            test.assertEqual(actual["requires_human"], expected["requires_human"])


class ScenarioTests(unittest.TestCase):
    def test_ten_shared_gold_cases_match_the_seed_records_and_policy(self):
        gold, reader = load_gold(), ScenarioReader()
        self.assertEqual(len(gold["cases"]), 10)
        self.assertEqual(len({case["id"] for case in gold["cases"]}), 10)
        clauses = {clause.id: clause for clause in load_corpus(ROOT / "policy/raw_policy/pdf")}
        for case in gold["cases"]:
            order = reader.read("orders", case["order_id"])
            self.assertEqual(order["customer_id"], case["customer_id"])
            self.assertLessEqual(set(case["item_ids"]), {item["id"] for item in reader.items_for_order(case["order_id"])})
            self.assertLessEqual(set(case["policy"]["clause_ids"] + case["policy"]["direct_reference_ids"]), set(clauses))
            ticket = reader.read("support_tickets", case["ticket_id"])
            self.assertEqual(ticket["order_id"], case["order_id"])
            self.assertEqual(ticket["customer_message"], case["message"])
            self.assertIn(case["order_id"], case["message"])
            self.assertNotIn("turns", case)
            self.assertNotEqual(case["message"], case["variants"]["typo"])
            self.assertNotEqual(case["message"], case["variants"]["paraphrase"])
            for architecture in gold["architectures"]:
                labels = case["architecture_expectations"][architecture]
                calls = case["required_reads"] + labels["policy_calls"] + labels["action_calls"]
                self.assertTrue(calls)
                self.assertEqual(labels["gold_tool_call_count"], len(calls))
                self.assertNotIn("optional_calls", labels)
                if architecture in {"s3", "s4"}:
                    self.assertEqual(calls[-1]["tool"], "route")
                    proposal = calls[-1]["arguments"]
                    self.assertIn("action", proposal)
                    self.assertEqual(proposal["route"], labels["outcome"]["expected_route"])
                    if case["expected"]["operation"] == "cancel":
                        self.assertEqual(proposal["reason_code"], "cancel")
                write = labels["outcome"]["write"]
                if write:
                    key = "record_id" if architecture in {"s1", "s2"} else "order_id"
                    self.assertEqual(write[key], case["order_id"])
            for name in case["policy"]["skill_files"]:
                self.assertTrue((ROOT / "policy/skills" / name).is_file())
            skill_text = "\n".join((ROOT / "policy/skills" / name).read_text()
                                   for name in case["policy"]["skill_files"])
            for clause_id in case["policy"]["clause_ids"]:
                self.assertIn(clause_id, skill_text)
                self.assertLessEqual(set(clauses[clause_id].references),
                                     set(case["policy"]["direct_reference_ids"]))
        check_gold(self, reader)

    def test_foreign_keys_quantities_and_paid_totals_are_consistent(self):
        reader = ScenarioReader()
        tables = reader.tables
        orders = {row["id"]: row for row in tables["orders"]}
        items = {row["id"]: row for row in tables["order_items"]}
        shipments = {row["id"]: row for row in tables["shipments"]}
        returns = {row["id"]: row for row in tables["returns"]}
        for case in load_gold()["cases"]:
            order_id = case["order_id"]
            order = orders[order_id]
            lines = [item for item in items.values() if item["order_id"] == order_id]
            self.assertEqual(sum(Decimal(item["paid_amount"]) for item in lines), Decimal(order["total"]))
            self.assertEqual(Decimal(order["subtotal"])+Decimal(order["tax"])+Decimal(order["shipping_fee"]), Decimal(order["total"]))
        for line in tables["return_items"]:
            self.assertEqual(returns[line["return_id"]]["order_id"], items[line["order_item_id"]]["order_id"])
            self.assertLessEqual(line["quantity"], items[line["order_item_id"]]["quantity"])
        for line in tables["shipment_items"]:
            self.assertEqual(shipments[line["shipment_id"]]["order_id"], items[line["order_item_id"]]["order_id"])
            self.assertLessEqual(line["quantity"], items[line["order_item_id"]]["quantity"])

    @unittest.skipUnless(os.environ.get("SUPPORT_OPS_TEST_DATABASE_URL"), "isolated PostgreSQL URL not configured")
    def test_real_postgres_reader_produces_the_same_gold_decisions(self):
        check_gold(self, PostgresReader(os.environ["SUPPORT_OPS_TEST_DATABASE_URL"]))
