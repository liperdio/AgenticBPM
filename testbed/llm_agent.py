"""Gemini-based invoice approval agent (Google AI Studio free tier).

Implements the same Agent interface as RuleBasedAgent (agent.py), so the
Orchestrator, Event Logger, and scoring logic need no changes to use this
instead. The frame (approval rules) is encoded in the system prompt rather
than hardcoded conditionals, letting the agent reason past superficial
noise (typos, rounding) that defeats exact-match rule checking.

The free tier caps requests per day (not context size, which is huge), so
decisions are fetched in batches via warm_up(cases) rather than one case
per API call. decide(case) then does a cache lookup, no change needed to
the Orchestrator's per-case loop.

Trade-off: cases resolved in the same batch share (near-)identical
timestamps, since they are decided together in one API call rather than
genuinely one after another. Per-case timing fidelity is reduced in
exchange for staying within the free tier's daily request cap.
"""

from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from agent import Decision
from case_generator import APPROVAL_LIMIT, Case

load_dotenv()  # reads .env in the project root, if present, into os.environ

FRAME_PROMPT = f"""You are an invoice approval agent for a company's accounts payable process.

For EACH invoice case given to you, decide whether to approve it directly or escalate it to a human reviewer.

Approve a case only if ALL of the following are true:
- A matching purchase order (PO) exists.
- The vendor on the invoice clearly matches the vendor on the PO. Minor typos or formatting differences that obviously refer to the same vendor still count as a match.
- The item description on the invoice clearly matches the item on the PO. Minor typos that obviously refer to the same item still count as a match.
- The invoice amount is strictly less than {APPROVAL_LIMIT}.

If a PO is missing, if the vendor is genuinely different (not just a typo), if the item is genuinely different, or if the amount is at or above {APPROVAL_LIMIT}, escalate that case instead of guessing.

You will be given multiple cases at once, each with a case_id. Call the submit_decisions
function exactly once with one decision entry per case_id given to you, in any order,
but every case_id must appear exactly once."""

BATCH_DECISION_FUNCTION = {
    "name": "submit_decisions",
    "description": "Submit approval decisions for a batch of invoice cases.",
    "parameters": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string"},
                        "decision": {
                            "type": "string",
                            "enum": ["approve", "escalate"],
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation, referencing the specific fields that drove the decision.",
                        },
                    },
                    "required": ["case_id", "decision", "reasoning"],
                },
            },
        },
        "required": ["decisions"],
    },
}


class GeminiAgent:
    name = "gemini_agent"

    # Free tier caps requests per day (not per minute) for this model; space
    # calls out and keep the batch count low to avoid burning the daily quota.
    MIN_SECONDS_BETWEEN_CALLS = 13.0
    DEFAULT_BATCH_SIZE = 20

    def __init__(self, model: str = "gemini-3.6-flash", batch_size: int = DEFAULT_BATCH_SIZE):
        self.model = model
        self.batch_size = batch_size
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get a free key from https://aistudio.google.com/apikey and set it before running this agent."
            )
        self.client = genai.Client(api_key=api_key)
        self._last_call_time: float | None = None
        self._cache: dict[str, Decision] = {}

        self._tools = types.Tool(function_declarations=[BATCH_DECISION_FUNCTION])
        self._config = types.GenerateContentConfig(
            system_instruction=FRAME_PROMPT,
            tools=[self._tools],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["submit_decisions"],
                )
            ),
        )

    def _case_to_prompt(self, case: Case) -> str:
        return (
            f"Case {case.case_id}:\n"
            f"- PO exists: {case.po_exists}\n"
            f"- Vendor on PO: {case.vendor_on_po}\n"
            f"- Vendor on invoice: {case.vendor_on_invoice}\n"
            f"- Item on PO: {case.item_on_po}\n"
            f"- Item on invoice: {case.item_on_invoice}\n"
            f"- PO amount: {case.po_amount}\n"
            f"- Invoice amount: {case.invoice_amount}\n"
        )

    def _throttle(self) -> None:
        if self._last_call_time is not None:
            elapsed = time.monotonic() - self._last_call_time
            wait = self.MIN_SECONDS_BETWEEN_CALLS - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_call_time = time.monotonic()

    def _call_with_retry(self, prompt: str, max_retries: int = 5):
        last_error = None
        for attempt in range(max_retries):
            self._throttle()
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self._config,
                )
            except errors.APIError as e:
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                if code not in (429, 503):
                    raise
                last_error = e
                time.sleep(max(10.0, 2**attempt))
        raise last_error

    def warm_up(self, cases: list[Case]) -> None:
        """Fetch decisions for all given cases in batches, caching them for decide()."""
        batches = [
            cases[i : i + self.batch_size] for i in range(0, len(cases), self.batch_size)
        ]
        print(f"[GeminiAgent] resolving {len(cases)} cases in {len(batches)} batched calls...")

        for batch_num, batch in enumerate(batches, start=1):
            prompt = "\n".join(self._case_to_prompt(c) for c in batch)
            response = self._call_with_retry(prompt)

            function_call = response.candidates[0].content.parts[0].function_call
            decisions = function_call.args["decisions"]

            usage = response.usage_metadata
            n = len(batch)
            per_case_input = (usage.prompt_token_count or 0) // n
            per_case_output = (usage.candidates_token_count or 0) // n

            returned_ids = set()
            for entry in decisions:
                case_id = entry["case_id"]
                returned_ids.add(case_id)
                self._cache[case_id] = Decision(
                    outcome=entry["decision"],
                    reasoning=entry["reasoning"],
                    model=self.model,
                    input_tokens=per_case_input,
                    output_tokens=per_case_output,
                )

            missing = {c.case_id for c in batch} - returned_ids
            if missing:
                print(f"[GeminiAgent] WARNING: batch {batch_num} missing decisions for: {missing}")

            print(f"[GeminiAgent] batch {batch_num}/{len(batches)} done ({len(returned_ids)}/{n} cases)")

    def decide(self, case: Case) -> Decision:
        if case.case_id not in self._cache:
            raise RuntimeError(
                f"No cached decision for {case.case_id}. Call warm_up(cases) with this case first."
            )
        return self._cache[case.case_id]
