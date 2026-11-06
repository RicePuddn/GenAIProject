"""Read the product and fulfilment lines for an order."""
from SupportOpsBot.services.structure.structure import GetOrderItemsInput, ItemsOutput
from SupportOpsBot.services.tools.tool import Tool


def create_tool(registry):
    return Tool("get_order_items", GetOrderItemsInput,
                "Get the order's item IDs, product names, SKU, quantity, price, and fulfilment status.",
                lambda args: {"items": registry.reader.items_for_order(args["order_id"])},
                output_model=ItemsOutput)

