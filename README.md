# Agentic Process Self-Improvement with Process Mining

IFN737/738 Research Project, School of Information Systems.
Supervisor: Adam Burke (at.burke@qut.edu.au)

## Overview

This project investigates how self-improving multi-agent systems can enhance
their own performance using process mining insights, under both autonomous
and human-in-the-loop (centaur) control.

**Research questions**

- **RQ1:** Can process mining identify actionable inefficiencies in
  multi-agent event logs, and how much do these insights improve process
  performance when fed back into the agent system?
- **RQ2:** How does autonomous self-modification versus human-in-the-loop
  intervention affect the magnitude, reliability, and safety of these
  improvements?

## Testbed

A minimal working case: a single LLM-based agent approves or escalates
invoices within a fixed frame (approve if under a threshold and details
match; otherwise escalate). See `testbed/case_generator.py`.

Planned pipeline: Case Generator -> Orchestrator -> Agent -> Event Logger ->
XES Event Log -> Process Mining -> Condition Switch (Baseline / Autonomous /
Centaur) -> feedback applied to the next batch of cases.

## Status

- [x] Literature review (6 papers analysed)
- [x] Research gap, problem, and RQs defined
- [x] Testbed architecture designed
- [x] Case generator built and validated (200 synthetic cases, 6 case types)
- [ ] Orchestrator, agent, and event logger
- [ ] Process mining step (PM4Py discovery / conformance checking)
- [ ] Autonomous and centaur feedback mechanisms
- [ ] Comparative evaluation

## Running the case generator

```bash
pip install -r requirements.txt
python testbed/case_generator.py
```

Generates 200 synthetic invoice cases (seeded for reproducibility) to
`testbed/data/cases.json`, with a known ground-truth outcome per case for
later evaluation.

## Repository structure

```
testbed/
  case_generator.py   # synthetic case generation
  data/                # generated output (not committed, see .gitignore)
```
