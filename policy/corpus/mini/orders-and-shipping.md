---
section: Orders and shipping
description: Mock policy for order status, shipment tracking, and order changes.
---
# Orders and shipping

## SHP-1 Tracking
Share only shipment status and estimated dates returned by the read tool. If a shipment is marked delayed or has no tracking event, explain that status plainly and offer human review when the customer needs a resolution.

## ORD-1 Order changes
Order changes are available only before fulfillment begins. Verify the current order status first. Any change that cannot be confirmed by the ticket-state tool must be described as a request pending review, not as completed.
