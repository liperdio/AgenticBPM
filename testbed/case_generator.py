"""Synthetic invoice case generator for the agentic BPMS testbed.

Produces invoice-approval cases with a known ground-truth outcome, so the
agent's decisions can be scored later. The approval frame is:
    approve if invoice_amount < APPROVAL_LIMIT and details match the PO;
    otherwise escalate.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

APPROVAL_LIMIT = 500.0

VENDORS = ["vendorID_0236", "vendorID_0427", "vendorID_0512", "vendorID_0089"]
ITEMS = [
    "Standard steel bracket",
    "Industrial cleaning solvent",
    "Replacement hydraulic hose",
    "Packaging tape, bulk",
    "Safety gloves, size L",
]

CASE_TYPES = (
    "clean_match",
    "minor_discrepancy",
    "boundary_amount",
    "invalid_vendor",
    "missing_po",
    "large_amount",
)


@dataclass
class Case:
    case_id: str
    case_type: str
    vendor_on_po: str
    vendor_on_invoice: str
    po_id: str
    po_amount: float
    invoice_amount: float
    item_on_po: str
    item_on_invoice: str
    po_exists: bool
    expected_outcome: str  # "approve" or "escalate", ground truth for evaluation


def _typo(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text
    i = rng.randrange(1, len(text) - 1)
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


def _expected_outcome(
    po_exists: bool,
    vendor_on_po: str,
    vendor_on_invoice: str,
    invoice_amount: float,
) -> str:
    if not po_exists:
        return "escalate"
    if vendor_on_po != vendor_on_invoice:
        return "escalate"
    if invoice_amount >= APPROVAL_LIMIT:
        return "escalate"
    return "approve"


def _base_case(case_id: str, case_type: str, rng: random.Random) -> tuple[str, str, str, str]:
    vendor = rng.choice(VENDORS)
    item = rng.choice(ITEMS)
    po_id = f"PO{rng.randint(100000, 999999)}"
    return vendor, item, po_id, case_id


def generate_case(case_id: str, case_type: str, rng: random.Random) -> Case:
    vendor, item, po_id, _ = _base_case(case_id, case_type, rng)
    po_amount = round(rng.uniform(20, 480), 2)

    vendor_on_invoice = vendor
    item_on_invoice = item
    invoice_amount = po_amount
    po_exists = True

    if case_type == "clean_match":
        pass  # everything matches exactly, amount already under the limit

    elif case_type == "minor_discrepancy":
        item_on_invoice = _typo(item, rng)
        invoice_amount = round(po_amount + rng.uniform(-0.5, 0.5), 2)

    elif case_type == "boundary_amount":
        po_amount = round(rng.uniform(490, 510), 2)
        invoice_amount = po_amount

    elif case_type == "invalid_vendor":
        others = [v for v in VENDORS if v != vendor]
        vendor_on_invoice = rng.choice(others)

    elif case_type == "missing_po":
        po_exists = False

    elif case_type == "large_amount":
        po_amount = round(rng.uniform(600, 5000), 2)
        invoice_amount = po_amount

    else:
        raise ValueError(f"unknown case_type: {case_type}")

    expected = _expected_outcome(po_exists, vendor, vendor_on_invoice, invoice_amount)

    return Case(
        case_id=case_id,
        case_type=case_type,
        vendor_on_po=vendor,
        vendor_on_invoice=vendor_on_invoice,
        po_id=po_id,
        po_amount=po_amount,
        invoice_amount=invoice_amount,
        item_on_po=item,
        item_on_invoice=item_on_invoice,
        po_exists=po_exists,
        expected_outcome=expected,
    )


def generate_batch(
    n: int,
    seed: int = 0,
    proportions: dict[str, float] | None = None,
) -> list[Case]:
    if proportions is None:
        proportions = {
            "clean_match": 0.35,
            "minor_discrepancy": 0.20,
            "boundary_amount": 0.15,
            "invalid_vendor": 0.10,
            "missing_po": 0.10,
            "large_amount": 0.10,
        }
    missing = set(CASE_TYPES) - set(proportions)
    if missing:
        raise ValueError(f"missing proportions for: {missing}")

    rng = random.Random(seed)
    weighted_types = list(proportions.keys())
    weights = list(proportions.values())

    cases = []
    for i in range(n):
        case_type = rng.choices(weighted_types, weights=weights, k=1)[0]
        case_id = f"case_{i:05d}"
        cases.append(generate_case(case_id, case_type, rng))
    return cases


def write_batch(cases: list[Case], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in cases], f, indent=2)


if __name__ == "__main__":
    batch = generate_batch(n=200, seed=42)
    write_batch(batch, "testbed/data/cases.json")

    by_type: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    for c in batch:
        by_type[c.case_type] = by_type.get(c.case_type, 0) + 1
        by_outcome[c.expected_outcome] = by_outcome.get(c.expected_outcome, 0) + 1

    print(f"Generated {len(batch)} cases -> testbed/data/cases.json\n")
    print("By case_type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k:<18} {v}")
    print("\nBy expected_outcome:")
    for k, v in sorted(by_outcome.items()):
        print(f"  {k:<18} {v}")
    print("\nSample case:")
    print(json.dumps(asdict(batch[0]), indent=2))
