"""Seed a compact, dedicated customer dataset for exercising every read/write tool.

Safe to rerun: inserts are idempotent and this script never truncates existing data.
Run from the repository root with ``.venv/Scripts/python.exe scripts/seed_tool_scenarios.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from SupportOpsBot.runtime.shared.config import ROOT, load_env

HOME = {"line1": "18 Orchard Avenue", "city": "Singapore", "postal_code": "238801", "country": "SG"}
TABLES = {
    "customers": ("id,first_name,last_name,email,phone,created_at,loyalty_tier,address", [
        ("CUS-TOOLS-001", "Alex", "Tan", "alex.tools@example.test", "+65 8000 9001", "2026-01-01T00:00:00+00:00", "standard", HOME),
    ]),
    "products": ("id,sku,name,category,unit_price,product_type", [
        ("PRD-TOOL-HP", "TOOL-HP-01", "Auralis Headphones", "Electronics", "189.00", "physical"),
        ("PRD-TOOL-WB", "TOOL-WB-01", "Terra Water Bottle", "Lifestyle", "32.00", "physical"),
        ("PRD-TOOL-LP", "TOOL-LP-01", "Luma Desk Lamp", "Home", "74.50", "physical"),
    ]),
    "orders": ("id,customer_id,placed_at,status,subtotal,shipping_fee,tax,total,currency,channel,shipping_address", [
        ("ORD-TOOLS-001", "CUS-TOOLS-001", "2026-10-30T09:00:00+00:00", "processing", "44.84", "0.00", "5.16", "20.00", "SGD", "web", HOME),
        ("ORD-TOOLS-002", "CUS-TOOLS-001", "2026-10-28T09:00:00+00:00", "shipped", "221.00", "5.00", "20.34", "246.34", "SGD", "web", HOME),
        ("ORD-TOOLS-003", "CUS-TOOLS-001", "2026-10-20T09:00:00+00:00", "delivered", "295.50", "5.00", "27.05", "327.55", "SGD", "web", HOME),
    ]),
    "order_items": ("id,order_id,product_id,quantity,unit_price,item_status,final_sale,paid_amount", [
        ("ITM-TOOLS-001-1", "ORD-TOOLS-001", "PRD-TOOL-HP", 1, "44.84", "open", False, "20.00"),
        ("ITM-TOOLS-002-1", "ORD-TOOLS-002", "PRD-TOOL-HP", 1, "189.00", "fulfilled", False, "210.00"),
        ("ITM-TOOLS-002-2", "ORD-TOOLS-002", "PRD-TOOL-WB", 1, "32.00", "fulfilled", False, "36.34"),
        ("ITM-TOOLS-003-1", "ORD-TOOLS-003", "PRD-TOOL-HP", 1, "189.00", "fulfilled", False, "211.46"),
        ("ITM-TOOLS-003-2", "ORD-TOOLS-003", "PRD-TOOL-WB", 1, "32.00", "fulfilled", False, "35.84"),
        ("ITM-TOOLS-003-3", "ORD-TOOLS-003", "PRD-TOOL-LP", 1, "74.50", "fulfilled", False, "80.25"),
    ]),
    "shipments": ("id,order_id,carrier,tracking_number,status,shipped_at,estimated_delivery,delivered_at,destination,label_created_at", [
        ("SHP-TOOLS-002", "ORD-TOOLS-002", "DHL", "TOOLS002SG", "in_transit", "2026-10-29T08:00:00+00:00", "2026-11-05", None, HOME, "2026-10-28T10:00:00+00:00"),
        ("SHP-TOOLS-003", "ORD-TOOLS-003", "SingPost", "TOOLS003SG", "delivered", "2026-10-21T08:00:00+00:00", "2026-10-25", "2026-10-24T09:00:00+00:00", HOME, "2026-10-20T10:00:00+00:00"),
    ]),
    "shipment_items": ("shipment_id,order_item_id,quantity", [
        ("SHP-TOOLS-002", "ITM-TOOLS-002-1", 1), ("SHP-TOOLS-002", "ITM-TOOLS-002-2", 1),
        ("SHP-TOOLS-003", "ITM-TOOLS-003-1", 1), ("SHP-TOOLS-003", "ITM-TOOLS-003-2", 1), ("SHP-TOOLS-003", "ITM-TOOLS-003-3", 1),
    ]),
    "shipment_events": ("id,shipment_id,occurred_at,status,location,details", [
        ("EVT-TOOLS-002-1", "SHP-TOOLS-002", "2026-10-28T11:00:00+00:00", "packed", "Origin warehouse", "Order packed and handed to dispatch."),
        ("EVT-TOOLS-002-2", "SHP-TOOLS-002", "2026-10-29T08:00:00+00:00", "carrier_pickup", "Origin country", "Dispatched to local carrier."),
        ("EVT-TOOLS-002-3", "SHP-TOOLS-002", "2026-10-30T19:00:00+00:00", "international_transit", "Airfreight", "Departed origin country by air."),
        ("EVT-TOOLS-002-4", "SHP-TOOLS-002", "2026-10-31T07:00:00+00:00", "customs_arrival", "Singapore", "Arrived in host country; awaiting local dispatch."),
        ("EVT-TOOLS-003-1", "SHP-TOOLS-003", "2026-10-20T12:00:00+00:00", "packed", "Origin warehouse", "Order packed."),
        ("EVT-TOOLS-003-2", "SHP-TOOLS-003", "2026-10-21T08:00:00+00:00", "carrier_pickup", "Singapore", "Collected by local carrier."),
        ("EVT-TOOLS-003-3", "SHP-TOOLS-003", "2026-10-24T09:00:00+00:00", "delivered", "Singapore", "Delivered to recipient."),
    ]),
    "payments": ("id,order_id,provider,method,status,amount,transaction_ref,paid_at", [
        ("PAY-TOOLS-001", "ORD-TOOLS-001", "Stripe", "Visa ending 4242", "authorized", "50.00", "TXN-TOOLS-001", None),
        ("PAY-TOOLS-002", "ORD-TOOLS-002", "Stripe", "Visa ending 4242", "captured", "246.34", "TXN-TOOLS-002", "2026-10-28T09:02:00+00:00"),
        ("PAY-TOOLS-003", "ORD-TOOLS-003", "Stripe", "Visa ending 4242", "captured", "327.55", "TXN-TOOLS-003", "2026-10-20T09:02:00+00:00"),
    ]),
    "returns": ("id,order_id,status,reason,requested_at,received_at,resolution,evidence_status,inspection_status,reviewed_at,authorized_at,approved_at,dispute_status,handed_over_at", [
        ("RET-TOOLS-003", "ORD-TOOLS-003", "approved", "faulty", "2026-10-25T09:00:00+00:00", "2026-10-27T09:00:00+00:00", "refund", "verified", "passed", "2026-10-28T09:00:00+00:00", "2026-10-25T09:00:00+00:00", "2026-10-28T09:00:00+00:00", "none", "2026-10-26T09:00:00+00:00"),
    ]),
    "return_items": ("return_id,order_item_id,quantity,unused,original_packaging,tags_complete,accessories_complete,approved_refund_amount", [
        ("RET-TOOLS-003", "ITM-TOOLS-003-2", 1, False, True, True, True, "30.00"),
    ]),
}


def seed() -> None:
    load_env()
    dsn = os.environ.get("SUPPORT_OPS_DATABASE_URL")
    if not dsn:
        raise RuntimeError("set SUPPORT_OPS_DATABASE_URL in .env")
    with psycopg.connect(dsn, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute((ROOT / "data/db/schema.sql").read_text())
        for table, (columns, rows) in TABLES.items():
            names = [part.strip() for part in columns.split(",")]
            updates = sql.SQL(",").join(
                sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(name), sql.Identifier(name))
                for name in names if name != "id")
            conflict = (sql.SQL(" ON CONFLICT DO NOTHING") if table in {"shipment_items", "return_items"} else
                        sql.SQL(" ON CONFLICT ({}) DO UPDATE SET {}").format(sql.Identifier("id"), updates))
            stmt = sql.SQL("INSERT INTO {} ({}) VALUES ({}){}").format(
                sql.Identifier(table), sql.SQL(",").join(map(sql.Identifier, names)),
                sql.SQL(",").join(sql.Placeholder() for _ in names), conflict)
            for row in rows:
                cur.execute(stmt, tuple(Jsonb(v) if isinstance(v, dict) else v for v in row))
    print("Seeded CUS-TOOLS-001 with three orders (1, 2, and 3 items) and read/write scenario records.")


if __name__ == "__main__":
    seed()
