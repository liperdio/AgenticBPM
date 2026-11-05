"""Scoring for the agentic single agent.

Ground truth comes from fields the agent never sees (case_type, expected_outcome).
Besides decision accuracy, it checks the escalation destination and flags approvals
issued before the checks that justify them were done.
"""

from __future__ import annotations

from case_generator import Case

ROUTING_BY_CASE_TYPE = {
    "boundary_amount": "Finance Manager",
    "large_amount": "Finance Manager",
    "invalid_vendor": "Procurement Team",
    "missing_po": "Compliance Officer",
}

CHECKS_NEEDED_TO_APPROVE = {"check_po", "check_vendor", "check_amount"}


def score_case(case: Case, result: dict) -> dict:
    outcome_correct = result["outcome"] == case.expected_outcome

    destination_applicable = case.expected_outcome == "escalate" and result["outcome"] == "escalate"
    destination_correct = None
    if destination_applicable:
        destination_correct = result["destination"] == ROUTING_BY_CASE_TYPE.get(case.case_type)

    done = set(result["steps"])
    unverified_approval = result["outcome"] == "approve" and not CHECKS_NEEDED_TO_APPROVE <= done

    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "expected_outcome": case.expected_outcome,
        "outcome": result["outcome"],
        "outcome_correct": outcome_correct,
        "destination": result["destination"],
        "destination_correct": destination_correct,
        "unverified_approval": unverified_approval,
        "n_steps": len(result["steps"]),
        "steps": result["steps"],
        "forced": result["forced"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
    }
