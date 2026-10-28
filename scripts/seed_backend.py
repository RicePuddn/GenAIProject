"""Populate the local support database with repeatable fictional ecommerce records."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
from support_ops.helpers.config import ROOT, load_env

BASE = datetime(2026, 9, 1, 9, tzinfo=timezone.utc)
FIRST = ["Avery", "Jordan", "Casey", "Morgan", "Taylor", "Riley", "Jamie", "Quinn", "Cameron", "Drew", "Skyler", "Reese", "Alex", "Robin", "Sam", "Emery", "Parker", "Rowan", "Blake", "Sage", "Finley", "Dakota", "Kendall", "Jules", "Shiloh"]
LAST = ["Tan", "Lim", "Wong", "Chen", "Patel", "Ng", "Koh", "Lee", "Raman", "Goh", "Teo", "Chua", "Ong", "Yeo", "Pillai", "Low", "Ho", "Sim", "Ang", "Lau", "Cruz", "Singh", "Nair", "Foo", "Boon"]
AREAS = [("Orchard", "238801"), ("Tampines", "529510"), ("Queenstown", "149053"), ("Woodlands", "738099"), ("Jurong East", "609601"), ("Toa Payoh", "319075"), ("Bishan", "570221"), ("Bedok", "460123")]
PRODUCTS = [
    ("PRD-1001", "AUR-ANC-01", "Auralis Noise-Cancelling Headphones", "Electronics", "189.00"),
    ("PRD-1002", "NOVA-PB-20", "Nova 20,000 mAh Power Bank", "Electronics", "59.90"),
    ("PRD-1003", "LUMA-LAMP-2", "Luma Adjustable Desk Lamp", "Home", "74.50"),
    ("PRD-1004", "TERRA-BTL-750", "Terra Insulated Bottle 750 ml", "Lifestyle", "32.00"),
    ("PRD-1005", "FLEX-MAT-6", "Flex Training Mat", "Fitness", "48.00"),
    ("PRD-1006", "PULSE-WATCH-3", "Pulse Fitness Watch", "Electronics", "229.00"),
    ("PRD-1007", "CLOUD-PILLOW-2", "Cloud Support Pillow", "Home", "65.00"),
    ("PRD-1008", "ROAM-BAG-28", "Roam Cabin Backpack 28L", "Travel", "96.00"),
    ("PRD-1009", "BREW-GRIND-1", "Brew Manual Coffee Grinder", "Kitchen", "54.90"),
    ("PRD-1010", "NEST-CHARGER-3", "Nest 3-Port USB-C Charger", "Electronics", "45.00"),
]
SCENARIOS = [
    ("delivery", "normal", "Where is my order? The tracking page has not updated since it left the sorting hub.", "Explain the latest carrier scan and estimated delivery."),
    ("delivery", "high", "My parcel was due yesterday and still has not arrived. Can you check what happened?", "Review carrier delay and offer next steps."),
    ("delivery", "normal", "The box arrived, but one item from my two-item order is missing.", "Check line items and partial shipment status."),
    ("delivery", "high", "The package says delivered, but nothing was at my door when I got home.", "Check delivery scan and route to carrier investigation."),
    ("returns", "normal", "The headphones are unused and still in their original packaging. Can I return them?", "Check delivery date and return eligibility."),
    ("returns", "normal", "I received a cracked desk lamp. I would like a replacement or refund.", "Record damage details and check resolution options."),
    ("refund", "normal", "The return was received last week, but I cannot see the refund on my card.", "Inspect return and payment refund state."),
    ("refund", "high", "Please refund the entire order today; the total was over the self-service limit.", "Verify amount and hand off for approval."),
    ("payment", "high", "I see two pending charges for this order. Was I charged twice?", "Compare payment transactions before advising."),
    ("order_change", "normal", "I typed the wrong apartment number. The order has not shipped yet.", "Check fulfillment state before requesting address change."),
    ("cancellation", "normal", "Please cancel this order. I placed it by mistake a few minutes ago.", "Verify order state and cancellation policy."),
    ("payment", "normal", "My payment failed, but the checkout page showed an error after I submitted it.", "Check payment status before requesting another attempt."),
    ("delivery", "normal", "Only one of the items arrived today. Is the rest coming separately?", "Check shipment and item-level fulfillment."),
    ("returns", "normal", "I changed my mind about the bottle. It is unopened and still in its original packaging.", "Check return window and item conditions."),
    ("refund", "normal", "I returned one item from the order. Will I get the full order amount back?", "Explain partial refund calculation from returned items."),
    ("delivery", "normal", "Can you confirm the delivery address for order ORD-2026-00018?", "Verify order and avoid exposing unrelated customer data."),
    ("product", "low", "Does the power bank in my order come with a USB-C cable?", "Use product details where available; do not invent."),
    ("account", "normal", "I have two orders under my account. Can you tell me which one is still processing?", "Use customer-scoped order data."),
    ("returns", "normal", "The return tracking says delivered to your warehouse. What happens next?", "Check return and refund progression."),
    ("delivery", "high", "The tracking says delivery attempted, but nobody rang the bell. Please help.", "Review carrier event and arrange human follow-up if needed."),
]


def insert_record(cur, table: str, **record) -> None:
    """Keep column names explicit and fixture inserts idempotent."""
    query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT (id) DO NOTHING").format(
        sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, record)),
        sql.SQL(", ").join(sql.Placeholder() for _ in record),
    )
    cur.execute(query, tuple(record.values()))


def seed_customers(cur) -> None:
    for i, first in enumerate(FIRST):
        last = LAST[(i * 7) % len(LAST)]
        area, postal = AREAS[i % len(AREAS)]
        address = {
            "line1": f"{18+i} {area} Avenue", "line2": f"#{i%20+1:02d}-{i%8+1:02d}",
            "city": "Singapore", "postal_code": postal, "country": "SG",
        }
        insert_record(
            cur, "customers", id=f"CUS-{1001+i}", first_name=first, last_name=last,
            email=f"{first.lower()}.{last.lower()}.{i+1}@example.test",
            phone=f"+65 8{i:03d} {1000+i:04d}", created_at=BASE-timedelta(days=500-i*9),
            loyalty_tier=("standard", "silver", "gold", "platinum")[i % 4], address=Jsonb(address),
        )


def seed_shipment(cur, i: int, order_id: str, status: str, placed: datetime, address: dict) -> None:
    if status in {"delivered", "partially_fulfilled"}:
        shipment_status = "delivered"
    elif status == "shipped":
        shipment_status = ("in_transit", "delayed", "delivery_attempted")[(i // 5) % 3]
    else:
        shipment_status = "cancelled" if status == "cancelled" else "label_created"
    shipped = placed + timedelta(days=1) if status not in {"processing", "cancelled"} else None
    delivered = placed + timedelta(days=4) if shipment_status == "delivered" else None
    shipment_id = f"SHP-{i+1:05d}"
    insert_record(
        cur, "shipments", id=shipment_id, order_id=order_id,
        carrier=("Ninja Van", "SingPost", "J&T Express")[i % 3], tracking_number=f"SG{2026000000+i}",
        status=shipment_status, shipped_at=shipped, estimated_delivery=(placed+timedelta(days=5)).date(),
        delivered_at=delivered, destination=Jsonb(address),
    )
    events = []
    if shipped:
        events.append((shipped, "picked_up", "Parcel collected from seller"))
    # The latest scan agrees with the shipment timestamps, including delivered_at.
    latest_at = delivered or (shipped + timedelta(days=1) if shipped else placed)
    events.append((latest_at, shipment_status, f"Parcel {shipment_status.replace('_', ' ')}"))
    for index, (occurred_at, event_status, details) in enumerate(events):
        insert_record(
            cur, "shipment_events", id=f"SEV-{i+1:05d}-{index+1}", shipment_id=shipment_id,
            occurred_at=occurred_at, status=event_status, location=AREAS[(i+index) % len(AREAS)][0],
            details=details,
        )


def seed_return(cur, i: int, order_id: str, placed: datetime, total: Decimal) -> None:
    status = ("requested", "in_transit", "received", "approved")[i % 4]
    requested = placed + timedelta(days=8)
    received = requested + timedelta(days=4) if status in {"received", "approved"} else None
    insert_record(
        cur, "returns", id=f"RET-{i+1:05d}", order_id=order_id, status=status,
        reason=("unsuitable fit", "damaged on arrival", "changed mind", "item not as described")[i % 4],
        requested_at=requested, received_at=received, resolution="refund" if status == "approved" else None,
    )
    if status == "approved":
        processed = (i // 20) % 2 == 0
        insert_record(
            cur, "refunds", id=f"RFD-{i+1:05d}", order_id=order_id, payment_id=f"PAY-{i+1:05d}",
            status="processed" if processed else "pending", amount=min(total, Decimal(PRODUCTS[i % 10][4])),
            reason="approved return", requested_at=received,
            processed_at=requested+timedelta(days=5) if processed else None,
            transaction_ref=f"RFN-SG-{800000+i}" if processed else None,
        )


def seed_orders(cur) -> None:
    for i in range(60):
        order_id = f"ORD-2026-{i+1:05d}"
        placed = BASE + timedelta(days=i % 48, hours=i % 12)
        status = ("delivered", "shipped", "processing", "cancelled", "partially_fulfilled")[i % 5]
        item_count = 2 if status == "partially_fulfilled" else 1 + (i % 3 == 0)
        chosen = [PRODUCTS[(i+j*3) % len(PRODUCTS)] for j in range(item_count)]
        quantities = [1 + (i+j) % 2 for j in range(item_count)]
        subtotal = sum(Decimal(product[4]) * quantity for product, quantity in zip(chosen, quantities))
        shipping = Decimal("0.00") if subtotal >= 100 else Decimal("5.00")
        tax = ((subtotal+shipping) * Decimal("0.09")).quantize(Decimal("0.01"))
        total = subtotal + shipping + tax
        area, postal = AREAS[i % len(AREAS)]
        address = {"line1": f"{20+i} {area} Road", "city": "Singapore", "postal_code": postal, "country": "SG"}
        insert_record(
            cur, "orders", id=order_id, customer_id=f"CUS-{1001+(i*7)%25}", placed_at=placed,
            status=status, subtotal=subtotal, shipping_fee=shipping, tax=tax, total=total,
            currency="SGD", channel=("web", "mobile_app", "marketplace")[i % 3], shipping_address=Jsonb(address),
        )
        for index, product in enumerate(chosen):
            fulfilled = status == "delivered" or (status == "partially_fulfilled" and index == 0)
            item_status = "fulfilled" if fulfilled else "cancelled" if status == "cancelled" else "open"
            insert_record(
                cur, "order_items", id=f"ITM-{i+1:04d}-{index+1}", order_id=order_id,
                product_id=product[0], quantity=quantities[index], unit_price=product[4], item_status=item_status,
            )
        seed_shipment(cur, i, order_id, status, placed, address)
        payment_status = "failed" if status == "cancelled" and i % 2 else "captured"
        insert_record(
            cur, "payments", id=f"PAY-{i+1:05d}", order_id=order_id, provider=("Stripe", "Adyen")[i % 2],
            method=("Visa ending 4242", "PayNow", "Mastercard ending 1881")[i % 3], status=payment_status,
            amount=total, transaction_ref=f"TXN-SG-{700000+i}", paid_at=placed if payment_status == "captured" else None,
        )
        if status == "delivered":
            seed_return(cur, i, order_id, placed, total)


def seed_tickets(cur) -> None:
    for i, (category, priority, message, resolution) in enumerate(SCENARIOS):
        order_no = (i * 3 + 2) % 60
        order_id = f"ORD-2026-{order_no+1:05d}"
        message = message.replace("ORD-2026-00018", order_id)
        order_placed = BASE + timedelta(days=order_no % 48, hours=order_no % 12)
        created_at = max(BASE+timedelta(days=40+i % 12), order_placed+timedelta(days=8))
        status = ("open", "in_progress", "resolved", "awaiting_customer")[i % 4]
        insert_record(
            cur, "support_tickets", id=f"TKT-2026-{i+1:04d}", customer_id=f"CUS-{1001+(order_no*7)%25}",
            order_id=order_id if category != "product" else None, created_at=created_at,
            channel=("email", "chat", "web_form")[i % 3], category=category, priority=priority, status=status,
            subject=message.split(".")[0][:100], customer_message=message,
            resolution=resolution if status == "resolved" else None,
            assigned_team=("order_support", "returns_team", "payments_team")[i % 3],
        )


def main() -> None:
    load_env()
    dsn = os.environ.get("SUPPORT_OPS_DATABASE_URL")
    if not dsn:
        raise RuntimeError("set SUPPORT_OPS_DATABASE_URL in .env (see .env.example)")
    with psycopg.connect(dsn, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute((ROOT / "data/backend_seed/schema.sql").read_text())
        for product_id, sku, name, category, price in PRODUCTS:
            insert_record(cur, "products", id=product_id, sku=sku, name=name, category=category, unit_price=price)
        seed_customers(cur)
        seed_orders(cur)
        seed_tickets(cur)
    print("Seeded deterministic synthetic dataset (25 customers, 10 products, 60 orders, 20 support tickets).")


if __name__ == "__main__":
    main()
