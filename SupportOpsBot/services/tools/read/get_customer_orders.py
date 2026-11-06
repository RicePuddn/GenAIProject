"""List the signed in customer's orders."""
from SupportOpsBot.services.structure.structure import CustomerOrdersOutput, GetCustomerOrdersInput
from SupportOpsBot.services.tools.tool import Tool


def create_tool(registry):
    def get_customer_orders(args):
        # The reader is constructed for this signed-in account; it cannot list another customer's records.
        if not hasattr(registry.reader, "own_orders"):
            raise ValueError("no customer is signed in, so their orders cannot be listed: "
                             "ask the customer for the order number")
        return {"orders": registry.reader.own_orders()}

    return Tool("get_customer_orders", GetCustomerOrdersInput,
                "List the signed-in customer's latest orders. Use this when no order ID was supplied.",
                get_customer_orders, output_model=CustomerOrdersOutput)

