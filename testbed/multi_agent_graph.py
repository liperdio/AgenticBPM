"""4-agent Procure-to-Pay process, built as a LangGraph state graph.

Agents (nodes): Intake -> Verification -> Approval -> (if escalated) Routing.
Each node calls a local Ollama model directly (same pattern as ollama_agent.py,
not the LangChain model wrapper) and logs paired start/complete events via the
existing EventLogger, so the resulting event log stays consistent with the
single-agent testbed's schema, just with more distinct `resource` values now.
"""

from __future__ import annotations

from typing import TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

from case_generator import APPROVAL_LIMIT, Case
from event_logger import EventLogger

MODEL = "llama3.1"


class ProcessState(TypedDict):
    case: Case
    logger: EventLogger
    intake_ok: bool
    intake_reason: str
    po_match: bool
    vendor_match: bool
    item_match: bool
    verification_reason: str
    approval_outcome: str  # "approve" or "escalate"
    approval_reason: str
    routing_destination: str
    routing_reason: str


def _chat(system_prompt: str, user_prompt: str, tool: dict, max_retries: int = 3) -> tuple[dict, int, int]:
    tokens_in = tokens_out = 0
    raw = None
    for _ in range(max_retries):
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=[tool],
        )
        tokens_in += response.get("prompt_eval_count", 0) or 0
        tokens_out += response.get("eval_count", 0) or 0
        tool_calls = response["message"].get("tool_calls")
        if tool_calls:
            return tool_calls[0]["function"]["arguments"], tokens_in, tokens_out
        raw = response["message"].get("content")
    raise RuntimeError(f"Model did not call the expected tool after {max_retries} tries. Raw: {raw!r}")


def _case_summary(case: Case) -> str:
    return (
        f"- PO exists: {case.po_exists}\n"
        f"- Vendor on PO: {case.vendor_on_po}\n"
        f"- Vendor on invoice: {case.vendor_on_invoice}\n"
        f"- Item on PO: {case.item_on_po}\n"
        f"- Item on invoice: {case.item_on_invoice}\n"
        f"- PO amount: {case.po_amount}\n"
        f"- Invoice amount: {case.invoice_amount}\n"
    )


# ---------------------------------------------------------------- Intake ----

INTAKE_PROMPT = "You check whether an invoice submission is complete enough to process. Call submit_intake with your result."
INTAKE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_intake",
        "description": "Submit the intake completeness check.",
        "parameters": {
            "type": "object",
            "properties": {
                "complete": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["complete", "reason"],
        },
    },
}


def intake_node(state: ProcessState) -> ProcessState:
    case = state["case"]
    state["logger"].log_start(case, "intake_agent")
    args, tok_in, tok_out = _chat(INTAKE_PROMPT, _case_summary(case), INTAKE_TOOL)
    state["intake_ok"] = args["complete"]
    state["intake_reason"] = args["reason"]
    state["logger"]._write(  # lifecycle "complete" event for this stage
        {
            "case_id": case.case_id,
            "concept_name": "intake_check",
            "lifecycle_transition": "complete",
            "timestamp": state["logger"]._now(),
            "resource": "intake_agent",
            "decision": "complete" if args["complete"] else "incomplete",
            "reasoning": args["reason"],
            "ai_input_tokens": tok_in,
            "ai_output_tokens": tok_out,
        }
    )
    return state


# ----------------------------------------------------------- Verification ----

VERIFICATION_PROMPT = (
    "You verify an invoice against its purchase order. Check PO existence, vendor match, and item match. "
    "Minor typos or formatting differences that obviously refer to the same vendor or item still count as a match. "
    "Call submit_verification with your findings."
)
VERIFICATION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_verification",
        "description": "Submit the verification result.",
        "parameters": {
            "type": "object",
            "properties": {
                "po_match": {"type": "boolean"},
                "vendor_match": {"type": "boolean"},
                "item_match": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["po_match", "vendor_match", "item_match", "reason"],
        },
    },
}


def verification_node(state: ProcessState) -> ProcessState:
    case = state["case"]
    state["logger"].log_start(case, "verification_agent")
    args, tok_in, tok_out = _chat(VERIFICATION_PROMPT, _case_summary(case), VERIFICATION_TOOL)
    state["po_match"] = args["po_match"]
    state["vendor_match"] = args["vendor_match"]
    state["item_match"] = args["item_match"]
    state["verification_reason"] = args["reason"]
    state["logger"]._write(
        {
            "case_id": case.case_id,
            "concept_name": "verification_check",
            "lifecycle_transition": "complete",
            "timestamp": state["logger"]._now(),
            "resource": "verification_agent",
            "decision": f"po={args['po_match']},vendor={args['vendor_match']},item={args['item_match']}",
            "reasoning": args["reason"],
            "ai_input_tokens": tok_in,
            "ai_output_tokens": tok_out,
        }
    )
    return state


# ---------------------------------------------------------------- Approval --

APPROVAL_PROMPT = f"""You make the final approve/escalate decision for an invoice, given verification results.

Approve only if the PO exists, the vendor matches, the item matches, AND the invoice amount is strictly less than {APPROVAL_LIMIT}.
Otherwise escalate. Call submit_decision with your decision and reasoning."""
APPROVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": "Submit the approval decision.",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["approve", "escalate"]},
                "reasoning": {"type": "string"},
            },
            "required": ["decision", "reasoning"],
        },
    },
}


def approval_node(state: ProcessState) -> ProcessState:
    case = state["case"]
    state["logger"].log_start(case, "approval_agent")
    prompt = (
        f"Verification findings: PO match={state['po_match']}, vendor match={state['vendor_match']}, "
        f"item match={state['item_match']} ({state['verification_reason']}).\n"
        f"Invoice amount: {case.invoice_amount}\n"
    )
    args, tok_in, tok_out = _chat(APPROVAL_PROMPT, prompt, APPROVAL_TOOL)
    state["approval_outcome"] = args["decision"]
    state["approval_reason"] = args["reasoning"]
    state["logger"]._write(
        {
            "case_id": case.case_id,
            "concept_name": "approval_decision",
            "lifecycle_transition": "complete",
            "timestamp": state["logger"]._now(),
            "resource": "approval_agent",
            "decision": args["decision"],
            "reasoning": args["reasoning"],
            "ai_input_tokens": tok_in,
            "ai_output_tokens": tok_out,
        }
    )
    return state


def route_after_approval(state: ProcessState) -> str:
    return "escalation_routing" if state["approval_outcome"] == "escalate" else END


# ------------------------------------------------------ Escalation Routing --

ROUTING_PROMPT = """You route an escalated invoice to the right team.

Choose exactly one destination:
- "Finance Manager" if the issue is about the invoice amount.
- "Procurement Team" if the issue is a vendor or item mismatch.
- "Compliance Officer" if the issue is a missing purchase order.

Call submit_routing with your destination and reasoning."""
ROUTING_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_routing",
        "description": "Submit the escalation routing destination.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "enum": ["Finance Manager", "Procurement Team", "Compliance Officer"],
                },
                "reasoning": {"type": "string"},
            },
            "required": ["destination", "reasoning"],
        },
    },
}


def escalation_routing_node(state: ProcessState) -> ProcessState:
    case = state["case"]
    state["logger"].log_start(case, "escalation_routing_agent")
    prompt = (
        f"Why this was escalated: {state['approval_reason']}\n"
        f"Verification detail: PO match={state['po_match']}, vendor match={state['vendor_match']}, "
        f"item match={state['item_match']}\n"
        f"Invoice amount: {case.invoice_amount}\n"
    )
    args, tok_in, tok_out = _chat(ROUTING_PROMPT, prompt, ROUTING_TOOL)
    state["routing_destination"] = args["destination"]
    state["routing_reason"] = args["reasoning"]
    state["logger"]._write(
        {
            "case_id": case.case_id,
            "concept_name": "escalation_routing",
            "lifecycle_transition": "complete",
            "timestamp": state["logger"]._now(),
            "resource": "escalation_routing_agent",
            "decision": args["destination"],
            "reasoning": args["reasoning"],
            "ai_input_tokens": tok_in,
            "ai_output_tokens": tok_out,
        }
    )
    return state


# ------------------------------------------------------------- Build graph --

def build_graph():
    graph = StateGraph(ProcessState)
    graph.add_node("intake", intake_node)
    graph.add_node("verification", verification_node)
    graph.add_node("approval", approval_node)
    graph.add_node("escalation_routing", escalation_routing_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "verification")
    graph.add_edge("verification", "approval")
    graph.add_conditional_edges("approval", route_after_approval, {"escalation_routing": "escalation_routing", END: END})
    graph.add_edge("escalation_routing", END)

    return graph.compile()


def run_case(app, case: Case, logger: EventLogger) -> ProcessState:
    initial_state: ProcessState = {
        "case": case,
        "logger": logger,
        "intake_ok": True,
        "intake_reason": "",
        "po_match": False,
        "vendor_match": False,
        "item_match": False,
        "verification_reason": "",
        "approval_outcome": "",
        "approval_reason": "",
        "routing_destination": "",
        "routing_reason": "",
    }
    return app.invoke(initial_state)
