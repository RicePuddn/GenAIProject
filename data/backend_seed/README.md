# Local ecommerce backend data

All records are deterministic and fictional. Tables model customers, products, orders,
order items, shipments, carrier events, payments, returns, refunds, and support tickets.
The generator includes delivered, delayed, processing, cancelled, and partially fulfilled
orders, failed payments, and returns at different stages.

From the repository root, configure `.env` as described in the [main README](../../README.md), then run:

```bash
docker compose up -d --wait postgres
.venv/bin/python scripts/seed_backend.py
```

The script creates missing tables from `schema.sql` and inserts missing fixture IDs in
one transaction. It preserves existing rows, so rerunning it does not apply generator
changes to records already present. There are 25 customers, 10 products, 60 orders,
60 shipments, 60 payments, 12 returns, 3 refunds, and 20 support tickets. Order item
and carrier event counts depend on fulfillment status.

Useful lookup IDs:

| Record | Example |
| --- | --- |
| Customer | `CUS-1001` |
| Delivered order | `ORD-2026-00001` (delivered 5 September 2026) |
| Shipment | `SHP-00001` (use the order ID with the shipment lookup tools) |
| Processing order | `ORD-2026-00003` |
| Cancelled order | `ORD-2026-00004` |
| Partially fulfilled order | `ORD-2026-00005` |
| Payment | `PAY-00001` |
| Return | `RET-00001` |
| Support ticket | `TKT-2026-0001` |

The named Docker volume preserves data across service restarts. To intentionally
remove all local records and regenerate fixtures, including generator corrections:

```bash
# Destructive: removes this Compose project's database volume.
docker compose down -v
docker compose up -d --wait postgres
.venv/bin/python scripts/seed_backend.py
```

The API reads these records but keeps mock approvals, audit logs, and handoffs in
request-local memory. It does not persist its ticket state into these tables.
