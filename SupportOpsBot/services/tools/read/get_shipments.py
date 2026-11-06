"""Read shipment records for an order."""
from SupportOpsBot.services.structure.structure import GetShipmentsInput, ShipmentsOutput
from SupportOpsBot.services.tools.tool import Tool


def create_tool(registry):
    return Tool("get_shipments", GetShipmentsInput, "Get every shipment for an order ID.",
                lambda args: {"shipments": registry.reader.shipments_for_order(args["order_id"])},
                output_model=ShipmentsOutput)

