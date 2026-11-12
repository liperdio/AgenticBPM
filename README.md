# Agentic Process Self-Improvement with Process Mining

IFN737/738 Research Project, School of Information Systems, QUT.

This repository is the working testbed for a research project on closing the loop between
process mining and LLM agents. It is shared so that students who continue this work do not
start from scratch: it records what was built, what was found, what did not work, and what
remains.

## Research questions

- **RQ1:** Can process mining identify actionable inefficiencies in multi-agent event logs,
  and how much do these insights improve process performance when fed back into the agent
  system?
- **RQ2:** How does autonomous self-modification versus human-in-the-loop (centaur)
  intervention affect the magnitude, reliability, and safety of these improvements?

## Status at a glance

| Area | Status |
|---|---|
| Literature review and research gap | Done (see the progress report) |
| Case generator (200 seeded synthetic invoices, 6 case types, known ground truth) | Done |
| Orchestrator, agent interface, lifecycle-aware event logger (XES-style, start and complete events) | Done |
| Single agent with tool calling on local models (llama3.2, qwen2.5, llama3.1) | Done, 5 complete 200-case runs |
| Process mining with PM4Py (discovery, conformance, variants) | Done for single-agent logs |
| Stochastic conformance with Ebi (earth movers', chi-squared, entropic relevance) | Done for single-agent logs |
| Coordinator with one-tool agents, no workflow engine | Pilot on 10 cases only |
| Full 200-case run of the coordinated version and mining of its traces | Not done (about 2.5 hours per run) |
| Feedback step (mined insight changes the agent) | Not started |
| Autonomous and centaur conditions, condition switch (RQ2) | Not started |
| BPIC event logs as a secondary dataset | Not started |

## What the experiments showed so far

Baseline results, with no feedback applied (single agent, 200 cases, 119 approve and 81 escalate):

| Model | Accuracy per complete run | Wrongly approved (latest run) | Earth movers' conformance to ground truth |
|---|---|---|---|
| llama3.2 (3B) | 66.5%, 67.5% | 62 | 0.705 |
| qwen2.5 (7B) | 70.0%, 69.5% | 0 | 0.695 |
| llama3.1 (8B) | 96.5% (one run) | 0 | 0.965 |

- The two smaller models score about 70% for opposite reasons. llama3.2 approves invoices that
  should be escalated (the unsafe direction), and qwen2.5 escalates clean invoices.
- About 95% of cases receive the same decision between two runs of the same model, so
  run-to-run noise is much smaller than the differences between models.
- With one decision per case, ordinary conformance checking is trivially perfect for every
  agent, so it cannot separate them. Stochastic conformance and per-case agreement can.
- Pilot (first 10 cases, llama3.1): the coordinator reached 9 of 10 correct decisions against
  7 of 10 for the single agent with check tools, and chose a different order of checks from case
  to case (7 distinct sequences). This shows the set-up works, not that it is more accurate.

## Decisions and lessons worth knowing

1. **No workflow engine.** No fixed workflow or graph is prescribed. The agents choose their own
   activities, and the process is mined from the event log afterwards.
2. **Local models, not cloud free tiers.** Free cloud quotas limit how many cases can be run, so
   all reported experiments use local models through Ollama. The Gemini agent (`llm_agent.py`)
   is kept for reference but is not part of the analyses.
3. **Tool-calling reliability varies.** phi4-mini and deepseek-r1:1.5b produced no valid tool
   calls in a 10-case test. Cases where a model never makes a valid call are recorded as
   `no_decision` so that runs stay complete.
4. **One activity per case limits process mining.** In the single-agent set-up the only activity
   is the decision, so discovery and control-flow conformance have little to find. Richer traces
   need agents that choose several activities, which is why the coordinated version exists.
5. **Case generator limitation.** In `missing_po` cases the generated input still lists PO vendor,
   item and amount values that match the invoice, which may weaken the missing-PO signal. It
   is worth fixing before further runs.
6. **Keep complete runs only.** Partial runs are written to separate `_test` logs so they never
   mix into the real logs. Use a different `run_id` for repeat runs.

## Getting started

Python 3.14 was used (earlier 3.10+ versions should work). Ebi was tested at `ebi-pm` 0.3.14.

```bash
pip install -r requirements.txt
```

Install Ollama, then pull the models:

```bash
ollama pull llama3.2
ollama pull qwen2.5
ollama pull llama3.1
```

Generate the cases (seeded, so the case set is identical on every machine):

```bash
python testbed/case_generator.py
```

Run a model over all 200 cases as a single agent (add a number to run only the first N cases):

```bash
python testbed/run_single_model.py llama3.1
python testbed/run_single_model.py llama3.1 10
```

Run the agentic single agent or the coordinator (arguments are N, model and run_id):

```bash
python testbed/run_agentic.py 10 llama3.1 1
python testbed/run_coordinated.py 10 llama3.1 1
```

Analyse the single-agent logs:

```bash
python testbed/process_mining_single.py
python testbed/instrumentation_single.py
python testbed/ebi_single.py
```

`testbed/data/` is generated output and is not committed. The cases can be regenerated, but the
event logs from the earlier runs are not in the repository, so rerun the models to reproduce the
tables above.

## Repository structure

```
testbed/
  case_generator.py          # seeded synthetic invoice cases with ground truth
  agent.py                   # Agent interface and a rule-based baseline
  orchestrator.py            # drives one case at a time and logs events
  event_logger.py            # paired start and complete events
  ollama_agent.py            # local single-decision agent (Ollama)
  llm_agent.py               # Gemini agent (free tier, kept for reference)
  agentic_agent.py           # single agent that chooses among check tools
  agentic_scoring.py         # scoring for the tool-using agent
  coordinated_agents.py      # coordinator plus one-tool agents, no workflow engine
  run_single_model.py        # run any Ollama model over the cases
  run_agentic.py             # run the tool-using agent
  run_coordinated.py         # run the coordinator
  run_ollama.py, run_qwen.py, run_llama31.py, run_gemini.py   # older per-model runners
  process_mining_single.py   # PM4Py pass over single-agent logs
  instrumentation_single.py  # tokens, time, conformance, stochastic conformance
  ebi_single.py              # Ebi stochastic conformance matrices
  process_mining_multi.py    # legacy: analysis of an earlier fixed four-stage pipeline's log
  data/                      # generated output (not committed)
```

## What remains

1. Complete three runs per model and recompute consistency.
2. Run the coordinated version on all 200 cases and mine the traces with PM4Py and Ebi.
3. Mine one error pattern (for example llama3.2 approving invalid vendors), change the agent
   frame, rerun and compare with the no-feedback baseline (RQ1).
4. Build the condition switch with the autonomous and centaur branches and compare them (RQ2).
5. Add BPIC logs as a secondary dataset.

## Related documents

The following are provided separately from this repository: the Assignment 2C research progress
report (full literature review, methodology and results tables), and the consultation slides.
