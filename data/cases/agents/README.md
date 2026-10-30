# Agent gold cases

[gold.json](gold.json) contains 10 standalone cases shared by S1 (single + RAG), S2
(single + skills), S3 (multi + RAG) and S4 (multi + skills). Seed records are in
[`scripts/seed_backend.py`](../../../scripts/seed_backend.py).

Use fresh workflow state and the fixed clock **31 October 2026, 09:00 UTC**.
The reference path is shared reads → policy calls → action calls. `gold_tool_call_count` includes
agent-visible tools; supervisor/QA calls and internal backend reads are separate.
Extra/repeated calls affect efficiency; final outcomes determine task success.
Gold labels stay with the evaluator. Typo/paraphrase variants are also included.

1. **Accepted SGD 20 refund** (`gold-01`)

    Order: `ORD-2026-00101`. My returned bottle was accepted. Can you process my refund for ORD-2026-00101?

    Shared reads (4): `read_record(orders)`, `get_items_for_order`, `get_returns_for_order`, `get_payments_for_order`.

    | Architecture | Policy tools | Action tool | Tool calls | Final outcome |
    | --- | --- | --- | ---: | --- |
    | S1 | `policy.search` | `ticket_state.write(refund)` | 6 | Record one simulated SGD 20 refund; resolve. |
    | S2 | `policy.open` → `policy.read(refunds.md)` | `ticket_state.write(refund)` | 7 | Record one simulated SGD 20 refund; resolve. |
    | S3 | `policy.search` | `route(refund, auto)` | 6 | Record one simulated SGD 20 refund; resolve. |
    | S4 | `policy.open` → `policy.read(refunds.md)` | `route(refund, auto)` | 7 | Record one simulated SGD 20 refund; resolve. |

    Policy evidence: `REF-Refunds-6.6`. Direct references are listed in JSON.

2. **Accepted SGD 60 refund** (`gold-02`)

    Order: `ORD-2026-00102`. The return for my power bank was accepted. Please refund ORD-2026-00102.

    Shared reads (4): `read_record(orders)`, `get_items_for_order`, `get_returns_for_order`, `get_payments_for_order`.

    | Architecture | Policy tools | Action tool | Tool calls | Final outcome |
    | --- | --- | --- | ---: | --- |
    | S1 | `policy.search` | `ticket_state.write(refund, review_reason)` | 6 | Single-agent review, then record SGD 60 refund. |
    | S2 | `policy.open` → `policy.read(refunds.md)` | `ticket_state.write(refund, review_reason)` | 7 | Single-agent review, then record SGD 60 refund. |
    | S3 | `policy.search` | `route(refund, approver)` | 6 | Approver/QA, then record SGD 60 refund. |
    | S4 | `policy.open` → `policy.read(refunds.md)` | `route(refund, approver)` | 7 | Approver/QA, then record SGD 60 refund. |

    Policy evidence: `REF-Refunds-6.6`. Direct references are listed in JSON.

3. **Accepted SGD 120 refund** (`gold-03`)

    Order: `ORD-2026-00103`. My headphones return was accepted. Please arrange the refund for ORD-2026-00103.

    Shared reads (4): `read_record(orders)`, `get_items_for_order`, `get_returns_for_order`, `get_payments_for_order`.

    | Architecture | Policy tools | Action tool | Tool calls | Final outcome |
    | --- | --- | --- | ---: | --- |
    | S1 | `policy.search` | `ticket_state.handoff` | 6 | Human handoff; no refund write. |
    | S2 | `policy.open` → `policy.read(refunds.md)` | `ticket_state.handoff` | 7 | Human handoff; no refund write. |
    | S3 | `policy.search` | `route(refund, human)` | 6 | Human handoff; no refund write. |
    | S4 | `policy.open` → `policy.read(refunds.md)` | `route(refund, human)` | 7 | Human handoff; no refund write. |

    Policy evidence: `REF-Refunds-6.6`. Direct references are listed in JSON.

4. **Disputed return inspection** (`gold-04`)

    Order: `ORD-2026-00104`. For ORD-2026-00104, the warehouse says I returned the wrong item, but I disagree. Please review it.

    Shared reads (4): `read_record(orders)`, `get_items_for_order`, `get_returns_for_order`, `get_payments_for_order`.

    | Architecture | Policy tools | Action tool | Tool calls | Final outcome |
    | --- | --- | --- | ---: | --- |
    | S1 | `policy.search` | `ticket_state.handoff` | 6 | Human handoff; no refund write. |
    | S2 | `policy.open` → `policy.read(eligibility.md)` | `ticket_state.handoff` | 7 | Human handoff; no refund write. |
    | S3 | `policy.search` | `route(none, human)` | 6 | Human handoff; no refund write. |
    | S4 | `policy.open` → `policy.read(eligibility.md)` | `route(none, human)` | 7 | Human handoff; no refund write. |

    Policy evidence: `RET-ReturnParcelDisputes-7.2`. Direct references are listed in JSON.

5. **Final-sale change-of-mind return** (`gold-05`)

    Order: `ORD-2026-00105`. I changed my mind about the unused final-sale bottle in ORD-2026-00105. Can I return it?

    Shared reads (4): `read_record(orders)`, `get_items_for_order`, `get_shipments_for_order`, `get_returns_for_order`.

    | Architecture | Policy tools | Action tool | Tool calls | Final outcome |
    | --- | --- | --- | ---: | --- |
    | S1 | `policy.search` | `policy.check(return_request)` | 6 | Decline the ordinary return; no write. |
    | S2 | `policy.open` → `policy.read(eligibility.md)` | `policy.check(return_request)` | 7 | Decline the ordinary return; no write. |
    | S3 | `policy.search` | `route(none, auto)` | 6 | Decline the ordinary return; no write. |
    | S4 | `policy.open` → `policy.read(eligibility.md)` | `route(none, auto)` | 7 | Decline the ordinary return; no write. |

    Policy evidence: `RET-ReturnEligibility-2.4`, `RET-ReturnEligibility-2.5`. Direct references are listed in JSON.

6. **Faulty final-sale item without its box** (`gold-06`)

    Order: `ORD-2026-00106`. The final-sale bottle in ORD-2026-00106 arrived faulty. I no longer have its original box. Can I return it?

    Shared reads (4): `read_record(orders)`, `get_items_for_order`, `get_shipments_for_order`, `get_returns_for_order`.

    | Architecture | Policy tools | Action tool | Tool calls | Final outcome |
    | --- | --- | --- | ---: | --- |
    | S1 | `policy.search` | `ticket_state.write(return_request)` | 6 | Record return authorization; no refund payment. |
    | S2 | `policy.open` → `policy.read(eligibility.md)` | `ticket_state.write(return_request)` | 7 | Record return authorization; no refund payment. |
    | S3 | `policy.search` | `route(return, auto)` | 6 | Record return authorization; no refund payment. |
    | S4 | `policy.open` → `policy.read(eligibility.md)` | `route(return, auto)` | 7 | Record return authorization; no refund payment. |

    Policy evidence: `RET-ReturnEligibility-2.5`, `RET-ReturnProcedure-4.1`. Direct references are listed in JSON.

7. **Unsafe evidence** (`gold-07`)

    Order: `ORD-2026-00107`. The battery in ORD-2026-00107 is leaking. I cannot safely switch it on to record a video. Can you help with my return?

    Shared reads (4): `read_record(orders)`, `get_items_for_order`, `get_shipments_for_order`, `get_returns_for_order`.

    | Architecture | Policy tools | Action tool | Tool calls | Final outcome |
    | --- | --- | --- | ---: | --- |
    | S1 | `policy.search` | `ticket_state.handoff` | 6 | Human handoff; do not reject for unsafe evidence. |
    | S2 | `policy.open` → `policy.read(eligibility.md)` | `ticket_state.handoff` | 7 | Human handoff; do not reject for unsafe evidence. |
    | S3 | `policy.search` | `route(none, human)` | 6 | Human handoff; do not reject for unsafe evidence. |
    | S4 | `policy.open` → `policy.read(eligibility.md)` | `route(none, human)` | 7 | Human handoff; do not reject for unsafe evidence. |

    Policy evidence: `RET-SupportingEvidence-3.2`. Direct references are listed in JSON.

8. **Return requested after 30 days** (`gold-08`)

    Order: `ORD-2026-00108`. I received ORD-2026-00108 35 days ago and have found a fault. Can I still return it?

    Shared reads (4): `read_record(orders)`, `get_items_for_order`, `get_shipments_for_order`, `get_returns_for_order`.

    | Architecture | Policy tools | Action tool | Tool calls | Final outcome |
    | --- | --- | --- | ---: | --- |
    | S1 | `policy.search` | `ticket_state.handoff` | 6 | Human handoff to assess an exception; no write. |
    | S2 | `policy.open` → `policy.read(eligibility.md)` | `ticket_state.handoff` | 7 | Human handoff to assess an exception; no write. |
    | S3 | `policy.search` | `route(none, human)` | 6 | Human handoff to assess an exception; no write. |
    | S4 | `policy.open` → `policy.read(eligibility.md)` | `route(none, human)` | 7 | Human handoff to assess an exception; no write. |

    Policy evidence: `RET-ReturnEligibility-2.6`, `RET-ReturnEligibility-2.1`. Direct references are listed in JSON.

9. **Cancellation before fulfillment** (`gold-09`)

    Order: `ORD-2026-00109`. Please cancel ORD-2026-00109. I placed it by mistake.

    Shared reads (3): `read_record(orders)`, `get_items_for_order`, `get_shipments_for_order`.

    | Architecture | Policy tools | Action tool | Tool calls | Final outcome |
    | --- | --- | --- | ---: | --- |
    | S1 | `policy.search` | `ticket_state.write(cancel)` | 5 | Record cancellation; no refund payment. |
    | S2 | `policy.open` → `policy.read(changes.md)` | `ticket_state.write(cancel)` | 6 | Record cancellation; no refund payment. |
    | S3 | `policy.search` | `route(order_change: cancel, auto)` | 5 | Record cancellation; no refund payment. |
    | S4 | `policy.open` → `policy.read(changes.md)` | `route(order_change: cancel, auto)` | 6 | Record cancellation; no refund payment. |

    Policy evidence: `ODR-OrderChangeCancel-4.1`. Direct references are listed in JSON.

10. **Split delivery with an overdue parcel** (`gold-10`)

    Order: `ORD-2026-00110`. Only one parcel from ORD-2026-00110 arrived. Where is the other parcel? It was due several days ago.

    Shared reads (4): `read_record(orders)`, `get_items_for_order`, `get_shipments_for_order`, `get_shipment_events_for_order`.

    | Architecture | Policy tools | Action tool | Tool calls | Final outcome |
    | --- | --- | --- | ---: | --- |
    | S1 | `policy.search` | `ticket_state.handoff` | 6 | Human handoff for the overdue parcel; no refund write. |
    | S2 | `policy.open` → `policy.read(tracking.md)` | `ticket_state.handoff` | 7 | Human handoff for the overdue parcel; no refund write. |
    | S3 | `policy.search` | `route(none, human)` | 6 | Human handoff for the overdue parcel; no refund write. |
    | S4 | `policy.open` → `policy.read(tracking.md)` | `route(none, human)` | 7 | Human handoff for the overdue parcel; no refund write. |

    Policy evidence: `SHP-SeparateShipPartialDel-6.2`, `SHP-DeliveryProblems-7.1`. Direct references are listed in JSON.

All writes above are simulated. Human-review cases succeed through the correct
handoff and customer explanation, without an automatic write.
