"""Coordinated multi-agent invoice approval: one tool per agent, no fixed workflow.

A coordinator agent reads the invoice and the findings so far, and dispatches one agent at a
time. The specialist agents each hold exactly one tool (po_agent, vendor_agent, item_agent,
amount_agent) and decision_agent holds the single `decide` tool that ends the case. Which agent
runs next, in what order, whether any are skipped or repeated, and when to decide are all the
coordinator's own choices, so the process is something to be mined, not prescribed.

Every model call is stateless (system prompt plus a fresh user message), which avoids the
multi-turn tool-call format problems small local models have.

Events: each specialist run is logged as an activity named after its tool (check_po, ...,
approve or escalate) with the agent as resource. Each coordinator dispatch is logged as a
separate "route" event with resource "coordinator", so coordination cost stays separable.
"""

from __future__ import annotations

import json

import ollama

from agentic_agent import DESTINATIONS, SYSTEM_PROMPT, TOOLS, _invoice_text, _run_tool
from case_generator import APPROVAL_LIMIT, Case
from event_logger import EventLogger

MAX_STEPS = 10

WORKER_TOOL = {
    "po_agent": "check_po",
    "vendor_agent": "check_vendor",
    "item_agent": "check_item",
    "amount_agent": "check_amount",
}
AGENTS = list(WORKER_TOOL) + ["decision_agent"]
_TOOL_SCHEMA = {t["function"]["name"]: t for t in TOOLS}

COORDINATOR_PROMPT = f"""You coordinate the review of one supplier invoice in a procure-to-pay process.
You cannot see the purchase order (PO) data yourself. You can dispatch these agents, one at a time:
- po_agent: looks up the purchase order for the invoice
- vendor_agent: compares the vendor on the invoice with the vendor on the PO
- item_agent: compares the item on the invoice with the item on the PO
- amount_agent: checks the invoice amount against the approval limit of {APPROVAL_LIMIT}
- decision_agent: makes the final approve or escalate decision and ends the case

A sound decision needs to know whether a PO exists, whether the vendor matches, and whether the
amount is under the limit. Item typos that clearly refer to the same item are tolerated.
You choose which agent to dispatch next, in what order, and when there is enough evidence to
dispatch decision_agent. Call dispatch with the agent name."""

DISPATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "dispatch",
        "description": "Dispatch one agent to work on the invoice.",
        "parameters": {
            "type": "object",
            "properties": {"agent": {"type": "string", "enum": AGENTS}},
            "required": ["agent"],
        },
    },
}

DECIDE_TOOL = {
    "type": "function",
    "function": {
        "name": "decide",
        "description": "Make the final decision on the invoice. This ends the case.",
        "parameters": {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": ["approve", "escalate"]},
                "reason": {"type": "string"},
                "destination": {"type": "string", "enum": DESTINATIONS},
            },
            "required": ["outcome", "reason"],
        },
    },
}

WORKER_PROMPT = {
    "po_agent": "You are the PO agent in a procure-to-pay process. Call your tool to look up the purchase order for this invoice.",
    "vendor_agent": "You are the vendor agent in a procure-to-pay process. Call your tool to compare the vendor on this invoice with the purchase order.",
    "item_agent": "You are the item agent in a procure-to-pay process. Call your tool to compare the item on this invoice with the purchase order.",
    "amount_agent": "You are the amount agent in a procure-to-pay process. Call your tool to check this invoice amount against the approval limit.",
}

FORCED_FINAL = {"outcome": "escalate", "destination": None}


def _findings_text(findings: list[tuple[str, dict]]) -> str:
    if not findings:
        return "Findings so far: none."
    return "Findings so far:\n" + "\n".join(f"- {agent}: {json.dumps(result)}" for agent, result in findings)


class CoordinatedAgents:
    def __init__(self, model: str = "llama3.1", max_steps: int = MAX_STEPS, max_retries: int = 3):
        self.model = model
        self.max_steps = max_steps
        self.max_retries = max_retries

    def _call(self, system: str, user: str, tool: dict) -> tuple[dict | None, int, int, str]:
        tokens_in = tokens_out = 0
        content = ""
        for _ in range(self.max_retries):
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                tools=[tool],
            )
            tokens_in += response.get("prompt_eval_count", 0) or 0
            tokens_out += response.get("eval_count", 0) or 0
            content = response["message"].get("content") or ""
            calls = response["message"].get("tool_calls")
            if calls:
                return {"name": calls[0]["function"]["name"], "arguments": calls[0]["function"]["arguments"] or {}}, tokens_in, tokens_out, content
        return None, tokens_in, tokens_out, content

    def _log(
        self, logger: EventLogger, case: Case, run_id: str, step: int, activity: str, resource: str,
        started: str, tin: int, tout: int, **extra,
    ) -> None:
        base = {"case_id": case.case_id, "run_id": run_id, "concept_name": activity, "resource": resource, "step": step}
        logger._write({**base, "lifecycle_transition": "start", "timestamp": started})
        logger._write(
            {**base, "lifecycle_transition": "complete", "timestamp": logger._now(), "ai_input_tokens": tin, "ai_output_tokens": tout, **extra}
        )

    def run_case(self, case: Case, logger: EventLogger, run_id: str = "1") -> dict:
        invoice = _invoice_text(case)
        findings: list[tuple[str, dict]] = []
        activities: list[str] = []
        routes: list[str] = []
        total_in = total_out = 0
        final: dict | None = None
        forced: str | None = None

        for step in range(1, self.max_steps + 1):
            started = logger._now()
            call, tin, tout, content = self._call(COORDINATOR_PROMPT, f"{invoice}\n{_findings_text(findings)}", DISPATCH_TOOL)
            total_in += tin
            total_out += tout
            if call is None:
                forced = "no_dispatch"
                break
            target = call["arguments"].get("agent")
            routes.append(str(target))
            self._log(logger, case, run_id, step, "route", "coordinator", started, tin, tout, dispatch_target=target, reasoning=content)
            if target not in AGENTS:
                findings.append(("coordinator", {"error": f"unknown agent {target}"}))
                continue

            started = logger._now()
            if target == "decision_agent":
                call, tin, tout, content = self._call(
                    SYSTEM_PROMPT + "\n\nA coordinator has gathered the findings below. Use them and call decide.",
                    f"{invoice}\n{_findings_text(findings)}",
                    DECIDE_TOOL,
                )
                total_in += tin
                total_out += tout
                args = call["arguments"] if call else {}
                if call is None or args.get("outcome") not in ("approve", "escalate"):
                    forced = "no_decision"
                    break
                destination = args.get("destination") if args["outcome"] == "escalate" else None
                if destination not in DESTINATIONS:
                    destination = None
                final = {"outcome": args["outcome"], "destination": destination, "reason": str(args.get("reason", ""))}
                activities.append(args["outcome"])
                self._log(
                    logger, case, run_id, step, args["outcome"], "decision_agent", started, tin, tout,
                    decision=final["outcome"], destination=final["destination"], reasoning=final["reason"],
                )
                break

            tool_name = WORKER_TOOL[target]
            call, tin, tout, content = self._call(WORKER_PROMPT[target] + "\n\n" + invoice, invoice, _TOOL_SCHEMA[tool_name])
            total_in += tin
            total_out += tout
            if call is None or call["name"] != tool_name:
                result = {"error": "agent did not use its tool"}
            else:
                result, _ = _run_tool(case, tool_name, {})
            findings.append((target, result))
            activities.append(tool_name)
            self._log(logger, case, run_id, step, tool_name, target, started, tin, tout, tool_result=result, reasoning=content)
        else:
            forced = "step_limit"

        if final is None:
            forced = forced or "step_limit"
            final = {**FORCED_FINAL, "reason": f"forced: {forced}"}
            activities.append("escalate")
            self._log(
                logger, case, run_id, len(routes) + 1, "escalate", "coordinator", logger._now(), 0, 0,
                decision="escalate", destination=None, forced=forced,
            )

        return {
            "case_id": case.case_id,
            "outcome": final["outcome"],
            "destination": final["destination"],
            "reason": final["reason"],
            "steps": activities,
            "routes": routes,
            "forced": forced,
            "tokens_in": total_in,
            "tokens_out": total_out,
        }
