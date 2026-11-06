"""Request independent approval for one exact write."""
from SupportOpsBot.services.structure.structure import AnyObjectOutput, RequestReviewArguments
from SupportOpsBot.services.tools.tool import Tool


def create_tool(registry):
    def request_review(args):
        checked = registry.writes.check(args)
        if checked["status"] != "allowed":
            return checked
        decision = checked["policy_check"]
        if decision["route"] != "approver":
            return {"status": "not_required", "policy_check": decision,
                    "reason": "This write needs no independent review; call its named write tool."}
        if refused := registry.writes.review(args, decision, record_approval=True):
            return refused
        return {"status": "approved", "policy_check": decision,
                "reason": "Approved for the same operation, order, amount and items; call the named write tool."}

    return Tool("request_review", RequestReviewArguments,
                "Request independent review for the same action, order, amount, and items you will write.",
                request_review, output_model=AnyObjectOutput)

