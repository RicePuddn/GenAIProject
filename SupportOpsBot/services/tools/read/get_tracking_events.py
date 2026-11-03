"""Read chronological carrier tracking events for an order."""
from SupportOpsBot.services.structure.structure import GetTrackingEventsInput, TrackingEventsOutput
from SupportOpsBot.services.tools.tool import Tool


def create_tool(registry):
    return Tool("get_tracking_events", GetTrackingEventsInput,
                "Get carrier tracking events for an order, in chronological order.",
                lambda args: {"events": registry.reader.shipment_events_for_order(args["order_id"])},
                output_model=TrackingEventsOutput)

