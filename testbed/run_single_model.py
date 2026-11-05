"""Run one local Ollama model over the cases as a single approve/escalate agent.

Usage (from the project root):
    python testbed/run_single_model.py <model> [N]

With N, only the first N cases run and the events go to a separate "_test" log so partial runs
never mix into the real log. Without N, all cases run and events are appended to the model's
log. A case where the model never makes a valid tool call is recorded as "no_decision"
instead of stopping the run, so every run stays complete.
"""

import sys

from agent import Decision
from event_logger import EventLogger
from ollama_agent import OllamaAgent
from orchestrator import load_cases, run, summarize


class SafeOllamaAgent(OllamaAgent):
    name = "ollama_agent"

    def decide(self, case, max_retries: int = 3) -> Decision:
        try:
            return super().decide(case, max_retries)
        except (RuntimeError, KeyError, TypeError) as exc:
            return Decision(outcome="no_decision", reasoning=str(exc)[:300], model=self.model)


model = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else None
tag = model.replace(":", "_").replace(".", "")

cases = load_cases("testbed/data/cases.json")
if n:
    cases = cases[:n]
    log_path = f"testbed/data/event_log_{tag}_test.jsonl"
else:
    log_path = f"testbed/data/event_log_{tag}.jsonl"

with EventLogger(log_path) as logger:
    results = run(cases, SafeOllamaAgent(model=model), logger)

summarize(results)
print(f"\nCases with no valid decision: {sum(r['actual_outcome'] == 'no_decision' for r in results)}")
