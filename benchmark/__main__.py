"""python -m benchmark

Writes the executive and technical reports from a real run of the twelve patients.
Rashi runs all three tracks. The other algorithms run Track A and Track B.
Track C stays unavailable for them.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.agent_eval import evaluate_scenarios
from benchmark.controller import execute_all, slim
from benchmark.optimize import run_pilot
from benchmark.reports import executive_report, technical_report

OUT = Path(__file__).resolve().parent / "reports"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = execute_all()
    slim_bundle = {
        "patient_count": bundle["patient_count"],
        "source_commit": bundle["source_commit"],
        "created_at": bundle["created_at"],
        "runs": [slim(run) for run in bundle["runs"]],
    }
    executive = executive_report(slim_bundle)
    technical = technical_report(slim_bundle)
    (OUT / "executive.md").write_text(executive, encoding="utf-8")
    (OUT / "technical.md").write_text(technical, encoding="utf-8")
    (OUT / "evidence.json").write_text(json.dumps(slim_bundle, indent=2, default=str), encoding="utf-8")
    (OUT / "critic.json").write_text(json.dumps(evaluate_scenarios(), indent=2), encoding="utf-8")
    pilot = run_pilot()
    (OUT / "pilot.json").write_text(json.dumps(pilot, indent=2), encoding="utf-8")
    print(f"Wrote reports to {OUT}")
    print(executive.splitlines()[0])


if __name__ == "__main__":
    main()
