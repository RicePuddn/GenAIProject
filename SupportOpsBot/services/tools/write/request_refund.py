"""Record a pending refund request through the shared policy gate."""
from SupportOpsBot.services.structure.structure import AnyObjectOutput, RequestRefundInput
from SupportOpsBot.services.tools.tool import Tool
from .common import perform_write


def create_tool(registry):
    return Tool("request_refund", RequestRefundInput,
                "Request a refund. This records a pending request; it does not move money.",
                lambda args: perform_write(registry, "refund", args), output_model=AnyObjectOutput)

