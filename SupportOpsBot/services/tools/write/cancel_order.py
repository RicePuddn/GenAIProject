"""Cancel an order through the shared policy gate."""
from SupportOpsBot.services.structure.structure import AnyObjectOutput, CancelOrderInput
from SupportOpsBot.services.tools.tool import Tool
from .common import perform_write


def create_tool(registry):
    return Tool("cancel_order", CancelOrderInput,
                "Request cancellation of an order. The shared rules and review gates decide the outcome.",
                lambda args: perform_write(registry, "cancel", args), output_model=AnyObjectOutput)

