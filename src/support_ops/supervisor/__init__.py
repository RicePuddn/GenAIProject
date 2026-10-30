"""The multi-agent supervisor: triage, delegation, reply drafting and escalation with a handoff packet.

supervisor.py  Supervisor: the LLM calls (triage, drafting the customer reply)
escalation.py  when to hand off after triage, the handoff packet, the fixed hold reply
outcome.py     what the reply may tell the customer, in customer terms
The graph that delegates to specialists and acts on their routes is systems/multi_agent.py.
"""
