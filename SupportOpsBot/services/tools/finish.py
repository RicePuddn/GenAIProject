"""Agentic control tool for finishing a run."""
from SupportOpsBot.services.structure.structure import FinishArguments, FinishOutput
from SupportOpsBot.services.tools.tool import Tool


def create_tool():
    return Tool("finish", FinishArguments, FinishArguments.__doc__, lambda args: {"status": "finished"},
                data=False, output_model=FinishOutput)

