"""Search or check policy using the configured RAG or skills adapter."""
from SupportOpsBot.services.structure.structure import AnyObjectOutput
from SupportOpsBot.services.tools.tool import Tool

CHECK_DESCRIPTION = (
    "check: the delivery rules for an order's parcels (lost, late or damaged in transit), with operation "
    "delivery and the order_id. Each named write tool applies the rules itself."
)
CHECK_NOTE = (
    "rule_sources name the sections this rule comes from, without their text: they cannot be cited. "
    "Cite clause IDs or files from a policy search or read."
)


def create_tool(registry, provider=None):
    provider = provider or registry.policy

    def search_or_check(args):
        if args["action"] != "check":
            return provider.run(args)
        # Delivery checks evaluate each parcel separately and return the strictest human requirement.
        parcels = registry.gate.facts.deliveries(args["order_id"])
        checks = [provider.run({**args, "facts": parcel}) for parcel in parcels or [{}]]
        decision = next((check for check in checks if check["requires_human"]), checks[0])
        return {**decision, "parcel_checks": checks, "note": CHECK_NOTE}

    return Tool("policy", provider.arguments, provider.description,
                search_or_check, output_model=AnyObjectOutput)

