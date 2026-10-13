"""First PM4Py pass over the single-agent event logs.

Each log is cleaned to one start/complete pair per case (logs were appended across
repeated runs), converted to a PM4Py event log, and compared against a reference
model discovered from the ground-truth outcomes. Prints one summary row per agent.

Usage (from the project root):
    python testbed/process_mining_single.py
"""

import json
import statistics
from collections import Counter
from datetime import datetime

import pandas as pd
import pm4py

CASES_PATH = "testbed/data/cases.json"
LOGS = {
    "rule_based": None,
    "llama3.2": "testbed/data/event_log_ollama.jsonl",
    "qwen2.5": "testbed/data/event_log_qwen.jsonl",
    "llama3.1": "testbed/data/event_log_llama31.jsonl",
    "gemini": "testbed/data/event_log_gemini_full.jsonl",
}

with open(CASES_PATH, encoding="utf-8") as f:
    GROUND_TRUTH = {c["case_id"]: c["expected_outcome"] for c in json.load(f)}


def load_clean(path: str) -> list[dict]:
    """Keep, per case, the last complete event and the last start event before it."""
    last_start: dict[str, dict] = {}
    kept: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            if ev["lifecycle_transition"] == "start":
                last_start[ev["case_id"]] = ev
            elif "decision" in ev and ev["case_id"] in last_start:
                kept[ev["case_id"]] = {"start": last_start[ev["case_id"]], "complete": ev}
    rows = []
    for case_id, pair in kept.items():
        s, c = pair["start"], pair["complete"]
        rows.append(
            {
                "case_id": case_id,
                "decision": c["decision"],
                "start_ts": datetime.fromisoformat(s["timestamp"]),
                "end_ts": datetime.fromisoformat(c["timestamp"]),
                "tokens_in": c.get("ai_input_tokens", 0),
                "tokens_out": c.get("ai_output_tokens", 0),
            }
        )
    return rows


def to_event_log(rows: list[dict]) -> pd.DataFrame:
    events = []
    for r in rows:
        activity = f"decide_{r['decision']}"
        events.append(
            {"case:concept:name": r["case_id"], "concept:name": activity, "lifecycle:transition": "start", "time:timestamp": r["start_ts"]}
        )
        events.append(
            {"case:concept:name": r["case_id"], "concept:name": activity, "lifecycle:transition": "complete", "time:timestamp": r["end_ts"]}
        )
    df = pd.DataFrame(events)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    return pm4py.format_dataframe(df, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")


def rule_based_rows() -> list[dict]:
    from agent import RuleBasedAgent
    from case_generator import Case

    agent = RuleBasedAgent()
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = [Case(**c) for c in json.load(f)]
    now = datetime.now()
    return [
        {"case_id": c.case_id, "decision": agent.decide(c).outcome, "start_ts": now, "end_ts": now, "tokens_in": 0, "tokens_out": 0}
        for c in cases
    ]


def main() -> None:
    reference_rows = [
        {"case_id": cid, "decision": outcome, "start_ts": datetime.now(), "end_ts": datetime.now(), "tokens_in": 0, "tokens_out": 0}
        for cid, outcome in GROUND_TRUTH.items()
    ]
    reference_log = to_event_log(reference_rows)
    ref_tree = pm4py.discover_process_tree_inductive(reference_log)
    ref_net, ref_im, ref_fm = pm4py.convert_to_petri_net(ref_tree)
    print("Reference model (discovered from ground-truth outcomes):")
    print(f"  {ref_tree}\n")

    summary = []
    for name, path in LOGS.items():
        rows = rule_based_rows() if path is None else load_clean(path)
        log = to_event_log(rows)

        correct = sum(1 for r in rows if GROUND_TRUTH[r["case_id"]] == r["decision"])
        fitness = pm4py.fitness_token_based_replay(log, ref_net, ref_im, ref_fm)["log_fitness"]
        precision = pm4py.precision_token_based_replay(log, ref_net, ref_im, ref_fm)
        latencies = [(r["end_ts"] - r["start_ts"]).total_seconds() for r in rows]

        summary.append(
            {
                "agent": name,
                "cases": len(rows),
                "accuracy": round(correct / len(rows), 3),
                "fitness_vs_reference": round(fitness, 3),
                "precision_vs_reference": round(precision, 3),
                "median_latency_s": round(statistics.median(latencies), 2),
                "total_tokens_in": sum(r["tokens_in"] for r in rows),
                "total_tokens_out": sum(r["tokens_out"] for r in rows),
                "approve_share": round(Counter(r["decision"] for r in rows)["approve"] / len(rows), 3),
            }
        )

    table = pd.DataFrame(summary)
    print(table.to_string(index=False))
    table.to_csv("testbed/data/pm_single_agent_summary.csv", index=False)
    print("\nSaved testbed/data/pm_single_agent_summary.csv")


if __name__ == "__main__":
    main()
