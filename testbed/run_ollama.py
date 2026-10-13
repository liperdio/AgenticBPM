"""Run the full case batch through OllamaAgent (local model) and print a summary.

Usage (from the project root):
    python testbed/run_ollama.py

Requires Ollama running locally with the model already pulled, e.g.:
    ollama pull llama3.2
"""

from ollama_agent import OllamaAgent
from orchestrator import load_cases, run, summarize
from event_logger import EventLogger

cases = load_cases("testbed/data/cases.json")
agent = OllamaAgent()

with EventLogger("testbed/data/event_log_ollama.jsonl") as logger:
    results = run(cases, agent, logger)

summarize(results)
