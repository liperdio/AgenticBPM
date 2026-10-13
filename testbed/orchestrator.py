"""Orchestrator: drives cases through an agent, one at a time, and logs
each decision as a pair of lifecycle-aware events.

Depends only on the Agent interface, not on any specific implementation,
so a real LLM-based agent can replace RuleBasedAgent later without any
change here.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent import Agent, RuleBasedAgent
from case_generator import Case
from event_logger import EventLogger


def load_cases(path: str | Path) -> list[Case]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [Case(**c) for c in raw]


def run(cases: list[Case], agent: Agent, logger: EventLogger, verbose: bool = True) -> list[dict]:
    results = []
    total = len(cases)
    for i, case in enumerate(cases, start=1):
        if verbose and (i == 1 or i % 10 == 0 or i == total):
            print(f"Processing case {i}/{total} ({case.case_id})...")
        logger.log_start(case, agent.name)
        decision = agent.decide(case)
        logger.log_complete(case, decision)
        results.append(
            {
                "case_id": case.case_id,
                "case_type": case.case_type,
                "expected_outcome": case.expected_outcome,
                "actual_outcome": decision.outcome,
                "correct": decision.outcome == case.expected_outcome,
            }
        )
    return results


def summarize(results: list[dict]) -> None:
    total = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"Ran {total} cases. Accuracy: {correct}/{total} ({correct / total:.1%})\n")

    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        t = r["case_type"]
        by_type.setdefault(t, {"total": 0, "correct": 0})
        by_type[t]["total"] += 1
        by_type[t]["correct"] += r["correct"]

    print("Accuracy by case type:")
    for t, counts in sorted(by_type.items()):
        acc = counts["correct"] / counts["total"]
        print(f"  {t:<18} {counts['correct']}/{counts['total']} ({acc:.1%})")

    print("\nMisclassified cases:")
    wrong = [r for r in results if not r["correct"]]
    if not wrong:
        print("  none")
    for r in wrong[:10]:
        print(f"  {r['case_id']} ({r['case_type']}): expected {r['expected_outcome']}, got {r['actual_outcome']}")


if __name__ == "__main__":
    cases = load_cases("testbed/data/cases.json")
    agent = RuleBasedAgent()

    with EventLogger("testbed/data/event_log.jsonl") as logger:
        results = run(cases, agent, logger)

    summarize(results)
