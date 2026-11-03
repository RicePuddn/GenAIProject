"""Create a return request through the shared policy gate."""
from SupportOpsBot.services.structure.structure import AnyObjectOutput, CreateReturnRequestInput
from SupportOpsBot.services.tools.tool import Tool
from .common import perform_write


def create_tool(registry):
    return Tool("create_return_request", CreateReturnRequestInput,
                "Request a return. The shared rules and review gates decide whether it can be recorded.",
                lambda args: perform_write(registry, "return_request", args), output_model=AnyObjectOutput)

