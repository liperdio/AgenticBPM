"""Run the first N cases through the 4-agent LangGraph and print per-agent accuracy.

Usage (from the project root):
    python testbed/run_multi_agent_sample.py [N]

N defaults to 10. Requires Ollama running locally with llama3.1 pulled.
"""

import sys

from event_logger import EventLogger
from multi_agent_graph import build_graph, run_case
from multi_agent_scoring import score_case, summarize
from orchestrator import load_cases

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

cases = load_cases("testbed/data/cases.json")[:n]
app = build_graph()

scores = []
with EventLogger("testbed/data/event_log_multi_agent_sample.jsonl") as logger:
    for i, case in enumerate(cases, start=1):
        print(f"Processing case {i}/{len(cases)} ({case.case_id})...")
        final_state = run_case(app, case, logger)
        scores.append(score_case(case, final_state))

print()
summarize(scores)
