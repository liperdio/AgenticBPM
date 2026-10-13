"""Per-agent accuracy scoring for the 4-agent Procure-to-Pay graph.

Ground truth for each agent is derived from fields the agents never see
(case_type, expected_outcome), the same fields case_generator.py stores
purely for evaluation. This module does not run any cases, it only scores
a (case, final_state) pair already produced by multi_agent_graph.run_case.
"""

from __future__ import annotations

from dataclasses import dataclass

from case_generator import Case

ROUTING_BY_CASE_TYPE = {
    "boundary_amount": "Finance Manager",
    "large_amount": "Finance Manager",
    "invalid_vendor": "Procurement Team",
    "missing_po": "Compliance Officer",
}


@dataclass
class CaseScore:
    case_id: str
    case_type: str
    intake_correct: bool
    po_match_correct: bool
    vendor_match_correct: bool
    item_match_correct: bool
    approval_correct: bool
    routing_applicable: bool
    routing_correct: bool | None
    end_to_end_correct: bool


def score_case(case: Case, final_state: dict) -> CaseScore:
    intake_correct = final_state["intake_ok"] is True

    expected_po_match = case.po_exists
    expected_vendor_match = case.case_type != "invalid_vendor"
    expected_item_match = True

    po_match_correct = final_state["po_match"] == expected_po_match
    vendor_match_correct = final_state["vendor_match"] == expected_vendor_match
    item_match_correct = final_state["item_match"] == expected_item_match

    approval_correct = final_state["approval_outcome"] == case.expected_outcome

    routing_applicable = case.expected_outcome == "escalate" and final_state["approval_outcome"] == "escalate"
    routing_correct = None
    if routing_applicable:
        expected_destination = ROUTING_BY_CASE_TYPE.get(case.case_type)
        routing_correct = expected_destination is not None and final_state["routing_destination"] == expected_destination

    if case.expected_outcome == "escalate":
        end_to_end_correct = approval_correct and bool(routing_correct)
    else:
        end_to_end_correct = approval_correct

    return CaseScore(
        case_id=case.case_id,
        case_type=case.case_type,
        intake_correct=intake_correct,
        po_match_correct=po_match_correct,
        vendor_match_correct=vendor_match_correct,
        item_match_correct=item_match_correct,
        approval_correct=approval_correct,
        routing_applicable=routing_applicable,
        routing_correct=routing_correct,
        end_to_end_correct=end_to_end_correct,
    )


def _rate(count: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{count}/{total} ({count / total:.1%})"


def summarize(scores: list[CaseScore]) -> None:
    total = len(scores)

    intake_hits = sum(s.intake_correct for s in scores)
    po_hits = sum(s.po_match_correct for s in scores)
    vendor_hits = sum(s.vendor_match_correct for s in scores)
    item_hits = sum(s.item_match_correct for s in scores)
    approval_hits = sum(s.approval_correct for s in scores)
    end_to_end_hits = sum(s.end_to_end_correct for s in scores)

    routing_cases = [s for s in scores if s.routing_applicable]
    routing_hits = sum(bool(s.routing_correct) for s in routing_cases)

    print(f"Scored {total} cases\n")
    print("Per-agent accuracy:")
    print(f"  Intake              {_rate(intake_hits, total)}")
    print(f"  Verification: po    {_rate(po_hits, total)}")
    print(f"  Verification: vendor{_rate(vendor_hits, total)}")
    print(f"  Verification: item  {_rate(item_hits, total)}")
    print(f"  Approval            {_rate(approval_hits, total)}")
    print(f"  Routing             {_rate(routing_hits, len(routing_cases))}")
    print(f"  End-to-end          {_rate(end_to_end_hits, total)}")

    print("\nEnd-to-end accuracy by case type:")
    by_type: dict[str, list[CaseScore]] = {}
    for s in scores:
        by_type.setdefault(s.case_type, []).append(s)
    for case_type in sorted(by_type):
        group = by_type[case_type]
        hits = sum(s.end_to_end_correct for s in group)
        print(f"  {case_type:<18} {_rate(hits, len(group))}")

    misses = [s for s in scores if not s.end_to_end_correct]
    if misses:
        print("\nEnd-to-end misses:")
        for s in misses:
            print(f"  {s.case_id} ({s.case_type})")
