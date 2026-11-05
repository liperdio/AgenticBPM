"""Single agentic invoice-approval agent: the model chooses which activities to perform.

The agent starts with only what is printed on the invoice. The purchase-order data is
hidden behind activity tools (check_po, check_vendor, check_item, check_amount), and the
case ends when the agent calls approve or escalate. There is no graph or workflow engine:
the order of activities, and whether any are skipped, comes from the model's own choices,
so the process is something to be mined from the event log, not something we prescribe.

Each tool call is logged as one activity with start and complete events. The start time is
when the model call that chose the tool began, so duration and tokens cover choosing plus
executing the activity.
"""

from __future__ import annotations

import json

import ollama

from case_generator import APPROVAL_LIMIT, Case
from event_logger import EventLogger

DESTINATIONS = ["Finance Manager", "Procurement Team", "Compliance Officer"]
MAX_STEPS = 8

SYSTEM_PROMPT = f"""You process one supplier invoice for a procure-to-pay process.

You can only see what is printed on the invoice. Information about the purchase order (PO) is
available only through your tools, and you decide which tools to call and in what order.

Decision rules:
- Approve only if a PO exists, the vendor on the invoice matches the vendor on the PO, and the
  invoice amount is strictly less than {APPROVAL_LIMIT}.
- Minor typos or formatting differences in the item description that clearly refer to the same
  item still count as a match.
- Otherwise escalate. When you escalate, give a short reason and choose one destination:
  Finance Manager if the problem is the invoice amount, Procurement Team if it is a vendor or
  item mismatch, Compliance Officer if there is no purchase order.

Call exactly one tool at a time. Finish the case by calling approve or escalate."""

FORMAT_REMINDER = "Respond with a tool call only: either the next check, or approve or escalate to finish."

_NO_ARGS = {"type": "object", "properties": {}}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_po",
            "description": "Look up the purchase order for this invoice. Returns whether a PO exists and its amount.",
            "parameters": _NO_ARGS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_vendor",
            "description": "Compare the vendor on the invoice with the vendor on the purchase order.",
            "parameters": _NO_ARGS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_item",
            "description": "Compare the item on the invoice with the item on the purchase order.",
            "parameters": _NO_ARGS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_amount",
            "description": "Check the invoice amount against the approval limit.",
            "parameters": _NO_ARGS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approve",
            "description": "Approve the invoice. This ends the case.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": "Escalate the invoice to a team. This ends the case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "destination": {"type": "string", "enum": DESTINATIONS},
                },
                "required": ["reason", "destination"],
            },
        },
    },
]


def _invoice_text(case: Case) -> str:
    return (
        f"Invoice {case.case_id}\n"
        f"- Vendor: {case.vendor_on_invoice}\n"
        f"- Item: {case.item_on_invoice}\n"
        f"- Amount: {case.invoice_amount}\n"
        f"- PO reference: {case.po_id}\n"
    )


def _run_tool(case: Case, name: str, args: dict) -> tuple[dict, dict | None]:
    """Execute a tool. Returns (result shown to the agent, final decision or None)."""
    if name == "check_po":
        if not case.po_exists:
            return {"po_found": False}, None
        return {"po_found": True, "po_amount": case.po_amount}, None
    if name == "check_vendor":
        if not case.po_exists:
            return {"error": "No purchase order on file."}, None
        return {
            "vendor_on_po": case.vendor_on_po,
            "vendor_on_invoice": case.vendor_on_invoice,
            "exact_match": case.vendor_on_po == case.vendor_on_invoice,
        }, None
    if name == "check_item":
        if not case.po_exists:
            return {"error": "No purchase order on file."}, None
        return {
            "item_on_po": case.item_on_po,
            "item_on_invoice": case.item_on_invoice,
            "exact_match": case.item_on_po == case.item_on_invoice,
        }, None
    if name == "check_amount":
        return {
            "invoice_amount": case.invoice_amount,
            "approval_limit": APPROVAL_LIMIT,
            "under_limit": case.invoice_amount < APPROVAL_LIMIT,
        }, None
    if name == "approve":
        return {"status": "approved"}, {"outcome": "approve", "destination": None, "reason": str(args.get("reason", ""))}
    if name == "escalate":
        destination = args.get("destination")
        if destination not in DESTINATIONS:
            return {"error": f"destination must be one of {DESTINATIONS}"}, None
        return {"status": "escalated"}, {"outcome": "escalate", "destination": destination, "reason": str(args.get("reason", ""))}
    return {"error": f"unknown tool {name}"}, None


class AgenticAgent:
    def __init__(self, model: str = "llama3.1", max_steps: int = MAX_STEPS, max_retries: int = 3):
        self.model = model
        self.max_steps = max_steps
        self.max_retries = max_retries

    def _call(self, messages: list[dict]) -> tuple[dict | None, int, int]:
        tokens_in = tokens_out = 0
        for _ in range(self.max_retries):
            response = ollama.chat(model=self.model, messages=messages, tools=TOOLS)
            tokens_in += response.get("prompt_eval_count", 0) or 0
            tokens_out += response.get("eval_count", 0) or 0
            if response["message"].get("tool_calls"):
                return response["message"], tokens_in, tokens_out
        return None, tokens_in, tokens_out

    def run_case(self, case: Case, logger: EventLogger, run_id: str = "1") -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _invoice_text(case)},
        ]
        steps: list[str] = []
        total_in = total_out = 0
        final: dict | None = None
        forced: str | None = None

        for step in range(1, self.max_steps + 1):
            started = logger._now()
            message, tin, tout = self._call(messages)
            total_in += tin
            total_out += tout
            if message is None:
                forced = "no_tool_call"
                break

            calls = message["tool_calls"]
            call = calls[0]
            name = call["function"]["name"]
            args = call["function"]["arguments"] or {}
            result, final = _run_tool(case, name, args)
            steps.append(name)
            self._log_activity(
                logger, case, run_id, step, name, started, args, result, tin, tout,
                reasoning=message.get("content") or "",
                final=final,
                dropped_calls=len(calls) - 1,
            )
            messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": [call]})
            messages.append({"role": "tool", "content": json.dumps(result), "tool_name": name})
            if final is not None:
                break
            messages.append({"role": "user", "content": FORMAT_REMINDER})
        else:
            forced = "step_limit"

        if final is None:
            forced = forced or "step_limit"
            final = {"outcome": "escalate", "destination": None, "reason": f"forced: {forced}"}
            started = logger._now()
            steps.append("escalate")
            self._log_activity(
                logger, case, run_id, len(steps), "escalate", started, {}, {"status": "forced"}, 0, 0,
                reasoning="", final=final, dropped_calls=0, forced=forced,
            )

        return {
            "case_id": case.case_id,
            "outcome": final["outcome"],
            "destination": final["destination"],
            "reason": final["reason"],
            "steps": steps,
            "forced": forced,
            "tokens_in": total_in,
            "tokens_out": total_out,
        }

    def _log_activity(
        self, logger: EventLogger, case: Case, run_id: str, step: int, name: str, started: str,
        args: dict, result: dict, tin: int, tout: int, reasoning: str, final: dict | None,
        dropped_calls: int, forced: str | None = None,
    ) -> None:
        base = {"case_id": case.case_id, "run_id": run_id, "concept_name": name, "resource": self.model, "step": step}
        logger._write({**base, "lifecycle_transition": "start", "timestamp": started})
        complete = {
            **base,
            "lifecycle_transition": "complete",
            "timestamp": logger._now(),
            "tool_args": args,
            "tool_result": result,
            "reasoning": reasoning,
            "ai_input_tokens": tin,
            "ai_output_tokens": tout,
            "parallel_calls_dropped": dropped_calls,
        }
        if final is not None:
            complete["decision"] = final["outcome"]
            complete["destination"] = final["destination"]
        if forced:
            complete["forced"] = forced
        logger._write(complete)
