"""Read one order by its order ID."""
from SupportOpsBot.services.structure.structure import AnyObjectOutput, GetOrderInput
from SupportOpsBot.services.tools.tool import Tool


def create_tool(registry):
    return Tool("get_order", GetOrderInput, "Get an order by its exact order ID.",
                lambda args: registry.reader.read("orders", args["order_id"]), output_model=AnyObjectOutput)

