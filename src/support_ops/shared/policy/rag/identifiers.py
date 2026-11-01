"""Names used when creating chunk IDs and storing the records in Pinecone."""

# Each policy's top-level section has a stable prefix; the PDF section number follows it.
POLICY_SECTION_PREFIXES = {
    "orders-and-shipping-policy.pdf": {
        "1": "ODR-Application",
        "2": "SHP-ShippingOptions",
        "3": "SHP-ShippingFees",
        "4": "ODR-OrderChangeCancel",
        "5": "ODR-OrderStatusTracking",
        "6": "SHP-SeparateShipPartialDel",
        "7": "SHP-DeliveryProblems",
    },
    "returns-and-refunds-policy.pdf": {
        "1": "RET-Application",
        "2": "RET-ReturnEligibility",
        "3": "RET-SupportingEvidence",
        "4": "RET-ReturnProcedure",
        "5": "RET-ShippingFees",
        "6": "REF-Refunds",
        "7": "RET-ReturnParcelDisputes",
    },
}


# Find the source PDF from a configured prefix and its original section number.
def policy_source(clause_id: str) -> str:
    prefix, separator, section_id = clause_id.rpartition("-")
    if separator:
        section = section_id.split(".")[0]
        for source, prefixes in POLICY_SECTION_PREFIXES.items():
            if prefixes.get(section) == prefix:
                return source
    raise ValueError(f"unknown policy clause ID: {clause_id}")
