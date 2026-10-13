"""Run the full case batch through OllamaAgent using qwen2.5 and print a summary.

Usage (from the project root):
    python testbed/run_qwen.py

Requires Ollama running locally with the model already pulled:
    ollama pull qwen2.5
"""

from ollama_agent import OllamaAgent
from orchestrator import load_cases, run, summarize
from event_logger import EventLogger

cases = load_cases("testbed/data/cases.json")
agent = OllamaAgent(model="qwen2.5")

with EventLogger("testbed/data/event_log_qwen.jsonl") as logger:
    results = run(cases, agent, logger)

summarize(results)
