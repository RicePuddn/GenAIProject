The two strategies, the same text for every agent that uses them: react (one step at a time) and plan_execute (plan up front, replan only on an error or for a missing step). plan_execute plans on two levels with one loop: the supervisor plans the parts of the message (`plan_parts`), a working agent plans its tool calls (`plan_calls`).

## react
Work one step at a time: make exactly one tool call per turn, and see its result before choosing the next.

## react_finish
When the work is done, call finish.

## react_stop
When you have what you need, stop: reply with a one-line summary and no tool call.

## nudge
Finish by calling the finish tool.

## plan_parts
Plan the parts of this message before any of them runs: one ask_specialist call per part, in the order they should be handled, each naming the specialist and exactly what it should handle. A specialist's own lookups and writes are its business, not part of your plan.

## plan_calls
Plan the tool calls you need for this case, in order, before any of them runs.

## plan
{level} Return JSON only: {"calls": [{"name": "TOOL_NAME", "arguments": {}}]}. Use only these tools: {tools}
{context}
Use literal known IDs and arguments; never invent the results of earlier calls. If a call needs a value you have not seen yet, such as the clause IDs to cite or an amount from the payments, end the plan after the calls that fetch it: once the plan has run, you are asked again with the results. Never use a placeholder for a value you have not seen. If previous_error is set, the last plan failed there: correct it using the evidence, and do not repeat calls whose results are already in it.
{end}

## plan_finish
End a plan with finish once nothing else is needed; finish can only be the last call.

## plan_stop
Return {"calls": []} when nothing is needed.

## plan_check_parts
The previous plan has run and each specialist's report is in the evidence. Plan only the parts still open, such as the refund a cancelled order is owed.

## plan_check_calls
The previous plan has run and its results are in the evidence. Plan only the calls still needed to complete the request, such as a refund owed after a cancellation.
