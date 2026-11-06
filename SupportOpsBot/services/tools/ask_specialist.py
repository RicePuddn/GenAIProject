"""Multi-agent control tool for delegating one part of a ticket."""
from SupportOpsBot.services.structure.structure import AskSpecialistArguments, AskSpecialistOutput
from SupportOpsBot.services.tools.tool import Tool


def create_tool(handler):
    return Tool("ask_specialist", AskSpecialistArguments, AskSpecialistArguments.__doc__, handler,
                data=False, output_model=AskSpecialistOutput)

