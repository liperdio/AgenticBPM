"""Lifecycle-aware event logger for the invoice-approval testbed.

Writes paired start/complete events per case, not a single timestamp,
so agent processing time and handover delay stay separable, the fix
motivated by Agent Miner's single-timestamp limitation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from agent import Decision
from case_generator import Case


class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write(self, event: dict) -> None:
        self._fh.write(json.dumps(event) + "\n")
        self._fh.flush()

    def log_start(self, case: Case, agent_name: str) -> None:
        self._write(
            {
                "case_id": case.case_id,
                "concept_name": "agent_decision",
                "lifecycle_transition": "start",
                "timestamp": self._now(),
                "resource": agent_name,
                "case_data": asdict(case),
            }
        )

    def log_complete(self, case: Case, decision: Decision) -> None:
        self._write(
            {
                "case_id": case.case_id,
                "concept_name": "agent_decision",
                "lifecycle_transition": "complete",
                "timestamp": self._now(),
                "resource": decision.model,
                "decision": decision.outcome,
                "reasoning": decision.reasoning,
                "ai_input_tokens": decision.input_tokens,
                "ai_output_tokens": decision.output_tokens,
            }
        )

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
