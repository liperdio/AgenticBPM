"""Run the full case batch through GeminiAgent and print a summary.

Usage (from the project root):
    python testbed/run_gemini.py
"""

from llm_agent import GeminiAgent
from orchestrator import load_cases, run, summarize
from event_logger import EventLogger

cases = load_cases("testbed/data/cases.json")
agent = GeminiAgent()
agent.warm_up(cases)

with EventLogger("testbed/data/event_log_gemini.jsonl") as logger:
    results = run(cases, agent, logger)

summarize(results)
