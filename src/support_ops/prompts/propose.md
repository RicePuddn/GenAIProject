The facts are gathered (evidence). Now propose the outcome as one JSON object, exactly these fields: {schema}

- outcome resolved: the request is answered from the evidence, or the writes carry out what the customer asked for; note what to tell the customer.
- outcome awaiting_customer: only the customer can supply a missing fact; note the question to ask.
- outcome awaiting_human: a person must decide; note why. Nothing is written.
- writes: the actions the customer asked for, in order: refund (recorded as pending; no money moves), return_request (with the order items coming back) or cancel, each with its order, amount (refunds), items, reason and cited_clauses. Cite the clause IDs or files a policy search or read returned; a policy check's rule sources cannot be cited. Leave writes empty for a question. The workflow checks each write against the rules, has it reviewed when that is required, then makes it.
- review_reason: how the retrieved policy and the recorded facts support the writes, for the independent reviewer.
- If feedback is set, the last proposal was refused for that reason: correct it, or answer the customer without the write.
