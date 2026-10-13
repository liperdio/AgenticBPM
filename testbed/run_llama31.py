"""Run the full case batch through OllamaAgent using llama3.1 (8B) and print a summary.

Usage (from the project root):
    python testbed/run_llama31.py

Requires Ollama running locally with the model already pulled:
    ollama pull llama3.1
"""

from ollama_agent import OllamaAgent
from orchestrator import load_cases, run, summarize
from event_logger import EventLogger

cases = load_cases("testbed/data/cases.json")
agent = OllamaAgent(model="llama3.1")

with EventLogger("testbed/data/event_log_llama31.jsonl") as logger:
    results = run(cases, agent, logger)

summarize(results)
