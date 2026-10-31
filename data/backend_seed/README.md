# Backend data

- `schema.sql`: tables, columns, constraints and indexes.
- [`scripts/seed_backend.py`](../../scripts/seed_backend.py): hardcoded rows grouped by table.
- [`gold.json`](../cases/agents/gold.json): 10 labelled cases linked to the seeded records.

```bash
docker compose up -d --wait postgres
.venv/bin/python scripts/seed_backend.py
```

The script uses `SUPPORT_OPS_DATABASE_URL` from `.env` and preserves existing rows.
Use a fresh database for schema changes: `CREATE TABLE IF NOT EXISTS` does not alter
existing tables. Changing `.env` does not reset an existing database password.

| Tables | Purpose |
| --- | --- |
| `customers`, `products`, `orders` | Customer, product and purchase records. |
| `order_items` | Quantities, fulfillment state, purchase-time final-sale status and paid line amounts. |
| `shipments`, `shipment_items`, `shipment_events` | Parcel dates, item allocation and carrier scans. |
| `payments`, `refunds` | Original captured payments and their completed/pending refunds. Refunds may link to a return. |
| `returns`, `return_items` | Request reason, affected goods, evidence review, inspection, disputes and approved amounts. |
| `support_tickets` | Customer messages and support status. |

Return authorization (`authorized_at`) permits dispatch; final acceptance
(`approved_at`) follows review. `handed_over_at` records carrier acceptance.
Extensions need a date and recorded approval. Nullable condition findings mean
unknown, not false.

Tools collect these records into temporary inputs for `Rules.check()`. Refund
checks use both the captured payment balance and the remaining return approval.
Missing evidence or approval facts cannot be supplied by the agent.

The seed contains 60 background orders for the demos plus the 10 gold orders
`ORD-2026-00101`–`ORD-2026-00110`, with tickets `TKT-EVAL-01`–`TKT-EVAL-10`.
See the [case overview](../cases/agents/README.md) for scenarios and expected results.
Gold cases use a fixed clock of 31 October 2026, 09:00 UTC.

All four systems write to these tables, and each write is recorded in `backend_writes`: S1 and S2
through the `backend_write` tool, S3 and S4 through the gateway. Conversations also save a row in `support_tickets` (multi-agent in code each turn, S1 and S2 when
the agent calls `update_ticket`, and on every handoff), under the signed-in `customer_id`, or `dummy_00` when none is sent. To go back to the seed rows, which deletes everything else:

```bash
.venv/bin/python scripts/seed_backend.py --reset
```

To test real database reads and writes, seed an isolated database and set
`SUPPORT_OPS_TEST_DATABASE_URL` before running `pytest`. The write tests delete all rows in that database.
