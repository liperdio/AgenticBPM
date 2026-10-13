"""Local, Ollama-based invoice approval agent (llama3.2 by default).

Implements the same Agent interface as RuleBasedAgent (agent.py), so the
Orchestrator, Event Logger, and scoring logic need no changes to use this
instead. The frame (approval rules) is encoded in the system prompt rather
than hardcoded conditionals, letting the agent reason past superficial
noise (typos, rounding) that defeats exact-match rule checking.

Unlike GeminiAgent, no rate limits or daily quotas apply here (everything
runs locally), so decisions are made one case at a time, restoring genuine
per-case timestamps in the event log rather than the batched approximation
used for the free-tier cloud agent.
"""

from __future__ import annotations

import ollama

from agent import Decision
from case_generator import APPROVAL_LIMIT, Case

FRAME_PROMPT = f"""You are an invoice approval agent for a company's accounts payable process.

Decide whether to approve an invoice directly or escalate it to a human reviewer.

Approve the invoice only if ALL of the following are true:
- A matching purchase order (PO) exists.
- The vendor on the invoice clearly matches the vendor on the PO. Minor typos or formatting differences that obviously refer to the same vendor still count as a match.
- The item description on the invoice clearly matches the item on the PO. Minor typos that obviously refer to the same item still count as a match.
- The invoice amount is strictly less than {APPROVAL_LIMIT}.

If a PO is missing, if the vendor is genuinely different (not just a typo), if the item is genuinely different, or if the amount is at or above {APPROVAL_LIMIT}, escalate instead of guessing.

Call the submit_decision function with your decision and a brief reasoning."""

DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": "Submit the approval decision for this invoice.",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["approve", "escalate"],
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation for the decision, referencing the specific fields that drove it.",
                },
            },
            "required": ["decision", "reasoning"],
        },
    },
}


class OllamaAgent:
    name = "ollama_agent"

    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def _case_to_prompt(self, case: Case) -> str:
        return (
            "Invoice case:\n"
            f"- PO exists: {case.po_exists}\n"
            f"- Vendor on PO: {case.vendor_on_po}\n"
            f"- Vendor on invoice: {case.vendor_on_invoice}\n"
            f"- Item on PO: {case.item_on_po}\n"
            f"- Item on invoice: {case.item_on_invoice}\n"
            f"- PO amount: {case.po_amount}\n"
            f"- Invoice amount: {case.invoice_amount}\n"
        )

    def decide(self, case: Case, max_retries: int = 3) -> Decision:
        response = None
        for attempt in range(max_retries):
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": FRAME_PROMPT},
                    {"role": "user", "content": self._case_to_prompt(case)},
                ],
                tools=[DECISION_TOOL],
            )
            if response["message"].get("tool_calls"):
                break
        else:
            raise RuntimeError(
                f"Model failed to call submit_decision for {case.case_id} after "
                f"{max_retries} attempts (smaller local models sometimes emit a "
                f"tool-call-shaped string as plain text instead of a real tool "
                f"call). Last raw response: {response['message'].get('content')!r}"
            )

        tool_calls = response["message"]["tool_calls"]
        args = tool_calls[0]["function"]["arguments"]
        return Decision(
            outcome=args["decision"],
            reasoning=args["reasoning"],
            model=self.model,
            input_tokens=response.get("prompt_eval_count", 0) or 0,
            output_tokens=response.get("eval_count", 0) or 0,
        )
