"""Agent interface for the invoice-approval testbed.

Any agent (rule-based, LLM-based, etc.) implements decide(case) -> Decision.
The orchestrator only depends on this interface, not on any specific
implementation, so a real LLM-based agent can be dropped in later without
changing the orchestrator or event logger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from case_generator import Case, APPROVAL_LIMIT


@dataclass
class Decision:
    outcome: str  # "approve" or "escalate"
    reasoning: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class Agent(Protocol):
    name: str

    def decide(self, case: Case) -> Decision: ...


class RuleBasedAgent:
    """A literal, exact-match implementation of the frame.

    Serves two purposes: (1) lets the orchestrator be built and tested
    today, before an LLM provider is chosen, and (2) is a genuine baseline
    to compare a future LLM-based agent against, since this agent cannot
    tolerate minor discrepancies (e.g. typos) the way an LLM should.
    """

    name = "rule_based_agent"

    def decide(self, case: Case) -> Decision:
        if not case.po_exists:
            return Decision(
                outcome="escalate",
                reasoning="No matching purchase order found.",
                model=self.name,
            )
        if case.vendor_on_po != case.vendor_on_invoice:
            return Decision(
                outcome="escalate",
                reasoning="Vendor on invoice does not exactly match vendor on PO.",
                model=self.name,
            )
        if case.invoice_amount >= APPROVAL_LIMIT:
            return Decision(
                outcome="escalate",
                reasoning=f"Invoice amount {case.invoice_amount} meets or exceeds the approval limit.",
                model=self.name,
            )
        if case.item_on_po != case.item_on_invoice:
            return Decision(
                outcome="escalate",
                reasoning="Item description on invoice does not exactly match PO.",
                model=self.name,
            )
        return Decision(
            outcome="approve",
            reasoning="Vendor, item, and amount match the PO within the approval limit.",
            model=self.name,
        )
