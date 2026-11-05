"""Stochastic conformance between single-agent logs with Ebi.

Each agent's log is reduced to one trace per case (the decision: approve or escalate), after
the same cleaning used in process_mining_single.py. The ground-truth outcomes are included as a
reference "agent". Prints three matrices (rows are the log, columns are the model):

- Earth mover's stochastic conformance (1 minus the Wasserstein distance; 1 is identical)
- Chi-squared stochastic conformance (a fraction; 1 is identical)
- Entropic relevance (bits per trace to encode the log under the model; lower is better)

Usage (from the project root):
    python testbed/ebi_single.py
"""

import math
import warnings

import ebi
import pandas as pd
import pm4py

from process_mining_single import GROUND_TRUTH, LOGS, load_clean, rule_based_rows

warnings.filterwarnings("ignore")

EXCLUDED = {"rule_based", "gemini"}


def to_event_log(decisions: dict[str, str]):
    base = pd.Timestamp("2026-01-01", tz="UTC")
    rows = [
        {"case:concept:name": cid, "concept:name": d, "time:timestamp": base + pd.Timedelta(seconds=i)}
        for i, (cid, d) in enumerate(decisions.items())
    ]
    df = pm4py.format_dataframe(
        pd.DataFrame(rows), case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp"
    )
    return pm4py.convert_to_event_log(df)


def fraction_value(result) -> float:
    return float(result[0])


def entropic_value(result) -> float:
    text = str(result).strip()
    for line in text.splitlines():
        if line.strip().startswith("Approximately"):
            return float(line.split()[-1])
    return float(eval(text.replace("log(", "math.log2("), {"math": math}))


def main() -> None:
    decisions = {"ground_truth": dict(GROUND_TRUTH)}
    for name, path in LOGS.items():
        if name in EXCLUDED:
            continue
        rows = rule_based_rows() if path is None else load_clean(path)
        decisions[name] = {r["case_id"]: r["decision"] for r in rows}

    logs = {name: to_event_log(d) for name, d in decisions.items()}
    names = list(logs)

    metrics = {
        "Earth mover's stochastic conformance (higher is closer)": lambda a, b: fraction_value(ebi.conformance_earth_movers(a, b)),
        "Chi-squared stochastic conformance (higher is closer)": lambda a, b: fraction_value(ebi.conformance_chi_squared(a, b)),
        "Entropic relevance, bits per trace (lower is closer; row = log, column = model)": lambda a, b: entropic_value(ebi.conformance_entropic_relevance(a, b)),
    }
    short = {0: "emsc", 1: "chi2", 2: "entropic"}

    for i, (title, fn) in enumerate(metrics.items()):
        matrix = pd.DataFrame(index=names, columns=names, dtype=float)
        for r in names:
            for c in names:
                matrix.loc[r, c] = round(fn(logs[r], logs[c]), 3)
        print(f"\n{title}")
        print(matrix.to_string())
        try:
            matrix.to_csv(f"testbed/data/ebi_single_agent_{short[i]}.csv")
        except PermissionError:
            print(f"(could not save ebi_single_agent_{short[i]}.csv, the file is open in another program)")

    print("\nApprove share per log:")
    for name, d in decisions.items():
        print(f"  {name:<14} {sum(v == 'approve' for v in d.values()) / len(d):.3f}")


if __name__ == "__main__":
    main()
