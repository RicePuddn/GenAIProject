"""Read payment records for an order."""
from SupportOpsBot.services.structure.structure import GetPaymentsInput, PaymentsOutput
from SupportOpsBot.services.tools.tool import Tool


def create_tool(registry):
    return Tool("get_payments", GetPaymentsInput, "Get the payments recorded for an order.",
                lambda args: {"payments": registry.reader.payments_for_order(args["order_id"])},
                output_model=PaymentsOutput)

