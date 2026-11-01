"""Each specialist's policy tool keeps to its own part of the policy unless a call says scope all; the
single agent's sees everything. It is the same provider either way, so citations and checks work alike."""
import pytest

from support_ops.setups.multi_common import POLICY_SCOPE
from support_ops.shared.policy.rag import RagPolicy
from support_ops.shared.policy.rag.policy import own_sections
from support_ops.shared.policy.skills import SkillsPolicy
from support_ops.shared.rules import Rules
from tests.fakes import ORDER, FakeRetriever, ScriptedLLM, services, text, tool_call
from tests.setups.scripts import run

RETURNS = POLICY_SCOPE["skills"]["returns_refunds"]


def test_skills_list_open_and_read_keep_to_the_specialist_s_folders():
    scoped = SkillsPolicy(rules=Rules.load()).scoped(RETURNS)
    assert scoped.run({"action": "list"})["skills"] == ["returns-and-refunds"]
    assert scoped.run({"action": "list", "scope": "all"})["skills"] == ["orders-and-shipping", "returns-and-refunds"]
    assert scoped.run({"action": "read", "skill": "returns-and-refunds", "file": "refunds.md"})["file"] == "refunds.md"
    with pytest.raises(ValueError, match="outside your policy skills"):
        scoped.run({"action": "open", "skill": "orders-and-shipping"})
    assert scoped.run({"action": "open", "skill": "orders-and-shipping", "scope": "all"})["file"] == "SKILL.md"
    with pytest.raises(ValueError, match="your skills: \\['returns-and-refunds'\\]"):
        scoped.run({"action": "open", "skill": "refunds"})   # a guessed name is offered only real, own skills
    assert "orders-and-shipping" not in scoped.planner_context()
    with pytest.raises(ValueError, match="do not exist"):
        SkillsPolicy(rules=Rules.load()).scoped(["returns"])   # a mistyped folder fails at once


def test_rag_search_keeps_hits_from_the_specialist_s_sections_and_their_references():
    scoped = RagPolicy(FakeRetriever(), Rules.load()).scoped(POLICY_SCOPE["rag"]["returns_refunds"])
    own = scoped.run({"action": "search", "query": "refund"})
    assert [clause["id"] for clause in own["clauses"]] == ["REF-Refunds-6.1"]
    assert (own["scope"], own["filtered_out"]) == ("own", 1)
    everything = scoped.run({"action": "search", "query": "refund", "scope": "all"})
    assert len(everything["clauses"]) == 2 and everything["scope"] == "all"
    referenced = {"clauses": [
        {"id": "A", "section": "Returns and Refunds Policy", "via": "search", "references": ["B"]},
        {"id": "B", "section": "Orders and Shipping Policy", "via": "reference"},
        {"id": "C", "section": "Orders and Shipping Policy", "via": "search"}]}
    assert [clause["id"] for clause in own_sections(referenced, ["Returns and refunds"])["clauses"]] == ["A", "B"]
    nothing = own_sections({"clauses": referenced["clauses"][2:]}, ["Returns and refunds"])
    assert "search again with scope all" in nothing["note"]


@pytest.mark.parametrize("mode", ["rag", "skills"])
def test_the_scoped_tool_runs_the_delivery_check_and_its_reads_can_be_cited(mode):
    tools = services(mode=mode)
    toolset = tools.registry.toolset(["policy", "backend_write"], policy=tools.policy.scoped(
        POLICY_SCOPE[mode]["returns_refunds"]))
    call = lambda name, **args: toolset.call(tools.ctx, "returns_refunds", tools.ctx.next_call_id(), name, args)
    assert "parcel_checks" in call("policy", action="check", operation="delivery", order_id=ORDER)
    if mode == "rag":
        call("policy", action="search", query="refund")
        cite = "REF-Refunds-6.1"
    else:
        call("policy", action="read", skill="returns-and-refunds", file="refunds.md")
        cite = "refunds.md"
    written = call("backend_write", operation="refund", order_id=ORDER, amount=20,
                   items=[{"item_id": "ITM-2", "quantity": 1}], reason="damaged_item", cited_clauses=[cite])
    assert written["status"] == "executed"


def test_specialists_get_the_scoped_tool_and_the_single_agent_does_not():
    for setup_id, role in (("s4-a", "specialist"), ("s8-a", "specialist"), ("s2-a", "agent")):
        llm = ScriptedLLM().push(role, tool_call("policy", action="list"))
        if role == "agent":
            llm.push(role, tool_call("finish", outcome="resolved", note="Listed."))
        else:
            llm.push("supervisor", tool_call("ask_specialist", specialist="returns_refunds", request="Refund it."))
            llm.push(role, tool_call("finish", outcome="resolved", note="Listed.") if setup_id == "s4-a"
                     else text("Gathered."))
            if setup_id == "s8-a":
                llm.push(role, text('{"outcome": "resolved", "note": "Listed."}'))
            llm.push("supervisor", tool_call("finish", outcome="resolved", note="Done.") if setup_id == "s4-a"
                     else text("Done."))
        output, _ = run(setup_id, llm)
        listed = output.evidence[0]["result"]["skills"]
        schema = next(s for _, _, schemas in llm.calls if schemas for s in schemas if s["function"]["name"] == "policy")
        system = [messages for called, messages, _ in llm.calls if called == role][0][0]["content"]
        if role == "agent":
            assert listed == ["orders-and-shipping", "returns-and-refunds"]
            assert "scope" not in schema["function"]["parameters"]["properties"]
            assert "scope all" not in system
        else:
            assert listed == ["returns-and-refunds"]
            assert "scope" in schema["function"]["parameters"]["properties"]
            assert "use scope all when the case crosses into another area" in system
