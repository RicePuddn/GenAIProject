"""Read return and refund records for an order."""
from SupportOpsBot.services.structure.structure import GetReturnsAndRefundsInput, ReturnsAndRefundsOutput
from SupportOpsBot.services.tools.tool import Tool


def create_tool(registry):
    return Tool("get_returns_and_refunds", GetReturnsAndRefundsInput,
                "Get the existing return requests and refunds recorded for an order.",
                lambda args: registry.reader.returns_for_order(args["order_id"]),
                output_model=ReturnsAndRefundsOutput)

