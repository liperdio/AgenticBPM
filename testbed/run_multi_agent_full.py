"""Run all cases through the 4-agent LangGraph, write the event log, and save per-case scores.

Usage (from the project root):
    python testbed/run_multi_agent_full.py [N]

N defaults to all cases. Requires Ollama running locally with llama3.1 pulled.
"""

import json
import sys
import time
from dataclasses import asdict

from event_logger import EventLogger
from multi_agent_graph import build_graph, run_case
from multi_agent_scoring import score_case, summarize
from orchestrator import load_cases

cases = load_cases("testbed/data/cases.json")
if len(sys.argv) > 1:
    cases = cases[: int(sys.argv[1])]

app = build_graph()
scores = []
start = time.time()

with EventLogger("testbed/data/event_log_multi_agent_full.jsonl") as logger:
    for i, case in enumerate(cases, start=1):
        final_state = run_case(app, case, logger)
        scores.append(score_case(case, final_state))
        if i % 10 == 0 or i == len(cases):
            print(f"Processed {i}/{len(cases)} cases ({time.time() - start:.0f}s elapsed)", flush=True)

with open("testbed/data/scores_multi_agent_full.json", "w", encoding="utf-8") as f:
    json.dump([asdict(s) for s in scores], f, indent=2)

print()
summarize(scores)
