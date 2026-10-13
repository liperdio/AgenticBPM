"""PM4Py pass over the 4-agent event log.

Discovers the as-executed process, checks conformance against (a) the designed
Procure-to-Pay model and (b) a ground-truth behavioural reference, and tabulates
time and token cost per activity.

Usage (from the project root):
    python testbed/process_mining_multi.py
"""

import json
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
import pm4py

LOG_PATH = "testbed/data/event_log_multi_agent_full.jsonl"
CASES_PATH = "testbed/data/cases.json"
OUT = "testbed/data/pm_multi_agent"

STAGES = ["intake_check", "verification_check", "approval_decision", "escalation_routing"]

with open(CASES_PATH, encoding="utf-8") as f:
    CASES = {c["case_id"]: c for c in json.load(f)}


def load_events() -> pd.DataFrame:
    starts: dict[tuple[str, str], datetime] = {}
    rows = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            key = (e["case_id"], e["resource"])
            ts = datetime.fromisoformat(e["timestamp"])
            if e["lifecycle_transition"] == "start":
                starts[key] = ts
            else:
                rows.append(
                    {
                        "case:concept:name": e["case_id"],
                        "concept:name": e["concept_name"],
                        "org:resource": e["resource"],
                        "start": starts[key],
                        "time:timestamp": ts,
                        "decision": e.get("decision", ""),
                        "tokens_in": e.get("ai_input_tokens", 0),
                        "tokens_out": e.get("ai_output_tokens", 0),
                    }
                )
    df = pd.DataFrame(rows)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["duration_s"] = (df["time:timestamp"] - df["start"]).dt.total_seconds()
    return df


def complete_log(df: pd.DataFrame) -> pd.DataFrame:
    return pm4py.format_dataframe(
        df[["case:concept:name", "concept:name", "time:timestamp"]].copy(),
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )


def ground_truth_log(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-01", tz="UTC")
    for cid in df["case:concept:name"].unique():
        path = STAGES[:3] + (["escalation_routing"] if CASES[cid]["expected_outcome"] == "escalate" else [])
        for i, act in enumerate(path):
            rows.append({"case:concept:name": cid, "concept:name": act, "time:timestamp": base + pd.Timedelta(seconds=i)})
    return pm4py.format_dataframe(
        pd.DataFrame(rows), case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp"
    )


def conformance(log: pd.DataFrame, net, im, fm) -> dict:
    out = {
        "fitness": round(pm4py.fitness_token_based_replay(log, net, im, fm)["log_fitness"], 3),
        "precision": round(pm4py.precision_token_based_replay(log, net, im, fm), 3),
    }
    try:
        from pm4py.algo.evaluation.simplicity import algorithm as simplicity

        out["simplicity"] = round(simplicity.apply(net), 3)
    except Exception:
        out["simplicity"] = None
    return out


def variant_distribution(log: pd.DataFrame) -> dict[tuple, float]:
    v = pm4py.get_variants(log)
    total = sum(v.values())
    return {k: n / total for k, n in v.items()}


def main() -> None:
    df = load_events()
    log = complete_log(df)
    n_cases = log["case:concept:name"].nunique()
    print(f"{n_cases} cases, {len(df)} activity instances\n")

    # as-executed model
    variants = pm4py.get_variants(log)
    variant_rows = [{"variant": " > ".join(k), "cases": n, "share": round(n / n_cases, 3)} for k, n in variants.items()]
    variant_table = pd.DataFrame(variant_rows).sort_values("cases", ascending=False)
    print("Variants (as executed):")
    print(variant_table.to_string(index=False), "\n")
    variant_table.to_csv(f"{OUT}_variants.csv", index=False)

    disc_net, disc_im, disc_fm = pm4py.discover_petri_net_inductive(log)
    print("Discovered model:", pm4py.discover_process_tree_inductive(log), "\n")

    # (a) designed model
    designed = pm4py.parse_process_tree(
        "->( 'intake_check', 'verification_check', 'approval_decision', X( tau, 'escalation_routing' ) )"
    )
    d_net, d_im, d_fm = pm4py.convert_to_petri_net(designed)
    # (b) ground-truth behavioural reference
    gt_log = ground_truth_log(df)
    g_net, g_im, g_fm = pm4py.discover_petri_net_inductive(gt_log)

    results = {
        "vs designed model": conformance(log, d_net, d_im, d_fm),
        "vs ground-truth reference": conformance(log, g_net, g_im, g_fm),
        "discovered model (self)": conformance(log, disc_net, disc_im, disc_fm),
    }
    print("Conformance of the agent log:")
    print(pd.DataFrame(results).T.to_string(), "\n")
    pd.DataFrame(results).T.to_csv(f"{OUT}_conformance.csv")

    # path agreement with ground truth, and a simple distribution distance
    actual = variant_distribution(log)
    truth = variant_distribution(gt_log)
    keys = set(actual) | set(truth)
    tvd = 0.5 * sum(abs(actual.get(k, 0) - truth.get(k, 0)) for k in keys)
    print(f"Total variation distance between actual and ground-truth variant distributions: {tvd:.3f}")
    print("(simple stand-in, Ebi stochastic metrics not yet computed)\n")

    path_of = df.groupby("case:concept:name")["concept:name"].apply(lambda s: "escalate" if "escalation_routing" in set(s) else "approve")
    confusion = pd.crosstab(
        pd.Series({c: CASES[c]["expected_outcome"] for c in path_of.index}, name="expected"),
        path_of.rename("actual path"),
    )
    print("Expected outcome vs path taken:")
    print(confusion.to_string(), "\n")

    # time and token cost per activity
    per_act = (
        df.groupby("concept:name")
        .agg(
            instances=("duration_s", "size"),
            median_s=("duration_s", "median"),
            mean_s=("duration_s", "mean"),
            tokens_in=("tokens_in", "sum"),
            tokens_out=("tokens_out", "sum"),
        )
        .reindex(STAGES)
        .round(2)
    )
    per_act["tokens_per_instance"] = ((per_act["tokens_in"] + per_act["tokens_out"]) / per_act["instances"]).round(0)
    print("Time and token cost per activity:")
    print(per_act.to_string(), "\n")
    per_act.to_csv(f"{OUT}_activity_cost.csv")

    case_time = df.groupby("case:concept:name").apply(
        lambda g: (g["time:timestamp"].max() - g["start"].min()).total_seconds(), include_groups=False
    )
    by_path = case_time.groupby(path_of).agg(["count", "median", "mean"]).round(2)
    print("Cycle time per case by path (s):")
    print(by_path.to_string())

    try:
        dfg, start, end = pm4py.discover_dfg(log)
        pm4py.save_vis_dfg(dfg, start, end, f"{OUT}_dfg.png")
        print(f"\nSaved {OUT}_dfg.png")
    except Exception as exc:
        print(f"\nDFG image not saved ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
