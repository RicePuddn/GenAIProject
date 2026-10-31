"""The single agent's scripted LLM: its independent checks approve unless a test scripts the approver."""
import json

from support_ops.helpers.llm import ScriptedLLM as BaseScriptedLLM, text

APPROVE = text(json.dumps({"decision": "approve", "issues": [], "reason": "Supported by the evidence."}))


class ScriptedLLM(BaseScriptedLLM):
    def __init__(self) -> None:
        super().__init__()
        self.on("approver", lambda *_: APPROVE)   # a pushed approver response is used first
