"""Run the coordinator plus one-tool-per-agent system over the cases, log every activity, and print a summary.

Usage (from the project root):
    python testbed/run_coordinated.py [N] [model] [run_id]

N defaults to all cases, model to llama3.1, run_id to 1. Use a different run_id for repeat
runs so they can be separated later in the shared log. Requires Ollama running locally.
"""

import json
import sys
import time
from collections import Counter

from coordinated_agents import CoordinatedAgents
from agentic_scoring import score_case
from event_logger import EventLogger
from orchestrator import load_cases

n = int(sys.argv[1]) if len(sys.argv) > 1 else None
model = sys.argv[2] if len(sys.argv) > 2 else "llama3.1"
run_id = sys.argv[3] if len(sys.argv) > 3 else "1"
tag = model.replace(":", "_").replace(".", "")

cases = load_cases("testbed/data/cases.json")
if n:
    cases = cases[:n]

agent = CoordinatedAgents(model=model)
log_path = f"testbed/data/event_log_coordinated_{tag}.jsonl"
scores = []
start = time.time()

with EventLogger(log_path) as logger:
    for i, case in enumerate(cases, start=1):
        result = agent.run_case(case, logger, run_id=run_id)
        scores.append(score_case(case, result))
        if i == 1 or i % 10 == 0 or i == len(cases):
            print(f"Processed {i}/{len(cases)} cases ({time.time() - start:.0f}s elapsed)", flush=True)

out_path = f"testbed/data/scores_coordinated_{tag}_run{run_id}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=2)

total = len(scores)
correct = sum(s["outcome_correct"] for s in scores)
print(f"\n{model}, run {run_id}: {total} cases, decision accuracy {correct}/{total} ({correct / total:.1%})")

dest = [s for s in scores if s["destination_correct"] is not None]
print(f"Escalation destination: {sum(s['destination_correct'] for s in dest)}/{len(dest)} correct")
print(f"Unverified approvals (approved before PO, vendor and amount were all checked): {sum(s['unverified_approval'] for s in scores)}")
print(f"Forced escalations (step limit or no tool call): {sum(1 for s in scores if s['forced'])}")
print(f"Mean steps per case: {sum(s['n_steps'] for s in scores) / total:.2f}")
print(f"Tokens: {sum(s['tokens_in'] for s in scores)} in, {sum(s['tokens_out'] for s in scores)} out")

print("\nAccuracy by case type:")
by_type: dict[str, list] = {}
for s in scores:
    by_type.setdefault(s["case_type"], []).append(s)
for t in sorted(by_type):
    g = by_type[t]
    print(f"  {t:<18} {sum(x['outcome_correct'] for x in g)}/{len(g)}")

print("\nActivity sequences (top 8):")
for seq, count in Counter(" > ".join(s["steps"]) for s in scores).most_common(8):
    print(f"  {count:>4}  {seq}")
