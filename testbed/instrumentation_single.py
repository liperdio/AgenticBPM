"""The four instrumentation items requested by the supervisor on the single-agent logs (llama3.2, qwen2.5, llama3.1).

1. Token cost per activity
2. Time per activity
3. Conformance (recall, precision, simplicity) of each agent log against the reference model
4. Stochastic conformance between agents with Ebi (earth mover's, chi-squared, entropic relevance)

In the single-agent setup the only activity is the decision, split here into decide_approve and
decide_escalate, so "per activity" means per decision outcome.

Usage (from the project root):
    python testbed/instrumentation_single.py
"""

import warnings

import pandas as pd
import pm4py
from pm4py.algo.evaluation.simplicity import algorithm as simplicity

import ebi_single
from process_mining_single import GROUND_TRUTH, LOGS, load_clean

warnings.filterwarnings("ignore")

AGENTS = {name: path for name, path in LOGS.items() if name not in ebi_single.EXCLUDED}


def to_log(rows: list[dict]) -> pd.DataFrame:
    events = []
    for r in rows:
        activity = f"decide_{r['decision']}"
        events.append({"case:concept:name": r["case_id"], "concept:name": activity, "time:timestamp": r["end_ts"]})
    df = pd.DataFrame(events)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    return pm4py.format_dataframe(df, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")


def main() -> None:
    data = {name: load_clean(path) for name, path in AGENTS.items()}

    # 1 and 2: token cost and time per activity
    rows = []
    for name, recs in data.items():
        for r in recs:
            rows.append(
                {
                    "agent": name,
                    "activity": f"decide_{r['decision']}",
                    "tokens_in": r["tokens_in"],
                    "tokens_out": r["tokens_out"],
                    "seconds": (r["end_ts"] - r["start_ts"]).total_seconds(),
                }
            )
    df = pd.DataFrame(rows)
    per_activity = df.groupby(["agent", "activity"]).agg(
        instances=("seconds", "size"),
        tokens_in_mean=("tokens_in", "mean"),
        tokens_out_mean=("tokens_out", "mean"),
        tokens_total=("tokens_in", "sum"),
        seconds_median=("seconds", "median"),
        seconds_mean=("seconds", "mean"),
        seconds_total=("seconds", "sum"),
    )
    per_activity["tokens_total"] = per_activity["tokens_total"] + df.groupby(["agent", "activity"])["tokens_out"].sum()
    print("1 and 2. Token cost and time per activity (single activity: the decision)")
    print(per_activity.round(2).to_string())
    per_agent = df.groupby("agent").agg(
        cases=("seconds", "size"),
        tokens_in=("tokens_in", "sum"),
        tokens_out=("tokens_out", "sum"),
        seconds_total=("seconds", "sum"),
        seconds_per_case=("seconds", "mean"),
    )
    print("\nTotals per agent")
    print(per_agent.round(2).to_string())
    per_activity.round(3).to_csv("testbed/data/instrumentation_single_per_activity.csv")

    # 3: conformance against the reference model discovered from the ground truth
    reference_rows = [{"case_id": cid, "decision": outcome, "end_ts": pd.Timestamp("2026-01-01", tz="UTC")} for cid, outcome in GROUND_TRUTH.items()]
    reference_log = to_log(reference_rows)
    ref_tree = pm4py.discover_process_tree_inductive(reference_log)
    ref_net, ref_im, ref_fm = pm4py.convert_to_petri_net(ref_tree)
    conf_rows = {}
    for name, recs in data.items():
        log = to_log(recs)
        disc_net, _, _ = pm4py.discover_petri_net_inductive(log)
        conf_rows[name] = {
            "recall (fitness)": round(pm4py.fitness_token_based_replay(log, ref_net, ref_im, ref_fm)["log_fitness"], 3),
            "precision": round(pm4py.precision_token_based_replay(log, ref_net, ref_im, ref_fm), 3),
            "simplicity of reference model": round(simplicity.apply(ref_net), 3),
            "simplicity of discovered model": round(simplicity.apply(disc_net), 3),
        }
    conf = pd.DataFrame(conf_rows).T
    print(f"\n3. Conformance against the reference model {ref_tree}")
    print(conf.to_string())
    conf.to_csv("testbed/data/instrumentation_single_conformance.csv")

    # 4: stochastic conformance between agents
    print("\n4. Stochastic conformance (Ebi)")
    ebi_single.main()


if __name__ == "__main__":
    main()
