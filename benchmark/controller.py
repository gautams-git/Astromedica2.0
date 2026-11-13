"""Central three-track controller.

One failed cell is stored and the remaining cells still run. Inputs are copied
per cell so a track cannot mutate the frozen patient.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable

from benchmark.adapters import run_track
from benchmark.architecture import ALGORITHM_MAP
from benchmark.compare import compare_tracks
from benchmark.hashing import git_commit, sha256_json
from benchmark.patients import all_patients, get_patient

ALGORITHMS = tuple(row["algorithm"] for row in ALGORITHM_MAP)
TRACKS = ("A", "B", "C")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def frozen_input(patient: dict) -> dict[str, Any]:
    chart = deepcopy(patient["chart"])
    rule1 = deepcopy(patient["rule1"])
    gender = chart.get("gender")
    snapshot = {
        "patient_id": patient["patient_id"],
        "case": patient["case"],
        "split": patient["split"],
        "gender": gender,
        "chart": chart,
        "rule1": rule1,
    }
    return {
        "snapshot": snapshot,
        "input_hash": sha256_json(snapshot),
        "schema_version": "benchmark-input-1",
        "missing": _missing(chart, rule1),
    }


def _missing(chart: dict, rule1: dict) -> list[str]:
    missing = []
    if not chart.get("gender"):
        missing.append("gender")
    if not (chart.get("ascendant") or {}).get("name"):
        missing.append("ascendant")
    if not chart.get("planets"):
        missing.append("planets")
    if not chart.get("dasha"):
        missing.append("dasha")
    if not rule1 or not any(rule1.get(key) for key in ("rogkaraka_1", "rogkaraka_3", "drishti_on_6th", "sixth_house_lord", "ascendant_lord", "sixth_house_sign")):
        missing.append("rule1")
    return missing


def _public_cell(cell: dict) -> dict:
    """Drop the raw engine blob from API/report copies. The ledger keeps it."""
    public = {key: value for key, value in cell.items() if key != "raw"}
    public["raw_preserved"] = cell.get("raw") is not None
    return public


def execute_patient(patient: dict, algorithms: Iterable[str] | None = None, tracks: Iterable[str] | None = None) -> dict[str, Any]:
    algorithms = tuple(algorithms or ALGORITHMS)
    tracks = tuple(tracks or TRACKS)
    frozen = frozen_input(patient)
    chart = frozen["snapshot"]["chart"]
    rule1 = frozen["snapshot"]["rule1"]
    gender = frozen["snapshot"]["gender"]
    cells = {}
    for algorithm in algorithms:
        cells[algorithm] = {}
        for track in tracks:
            cell = run_track(algorithm, track, chart, rule1, gender)
            cell["input_hash"] = frozen["input_hash"]
            cells[algorithm][track] = cell

    differentials = {}
    for algorithm, tracks_map in cells.items():
        differentials[algorithm] = {
            "A_vs_B": compare_tracks(tracks_map.get("A"), tracks_map.get("B"), "A", "B"),
            "B_vs_C": compare_tracks(tracks_map.get("B"), tracks_map.get("C"), "B", "C"),
            "A_vs_C": compare_tracks(tracks_map.get("A"), tracks_map.get("C"), "A", "C"),
        }

    record = {
        "benchmark_id": "triad5-rashi",
        "run_id": uuid.uuid4().hex,
        "patient_id": patient["patient_id"],
        "case": patient["case"],
        "split": patient["split"],
        "created_at": _now(),
        "source_commit": git_commit(),
        "input": frozen,
        "configuration_hash": sha256_json({"algorithms": algorithms, "tracks": tracks, "benchmark": "triad5-rashi-1"}),
        "cells": cells,
        "differentials": differentials,
        "limitations": [
            "Track C LLM weight selection is implemented only for Rashi Correlation.",
            "No clinical accuracy claim is made. The inputs are saved charts.",
            "Verification is limited wherever an engine does not expose contribution trails.",
        ],
    }
    record["record_hash"] = sha256_json({
        "patient_id": record["patient_id"],
        "input_hash": frozen["input_hash"],
        "configuration_hash": record["configuration_hash"],
        "tops": {
            algorithm: {
                track: [item.get("label") for item in (cell.get("rankings") or [])[:5]]
                for track, cell in tracks_map.items()
            }
            for algorithm, tracks_map in cells.items()
        },
    })
    return record


def execute_all(patient_ids: Iterable[str] | None = None, **kwargs) -> dict[str, Any]:
    patients = all_patients()
    if patient_ids is not None:
        wanted = set(patient_ids)
        patients = [item for item in patients if item["patient_id"] in wanted]
    runs = [execute_patient(patient, **kwargs) for patient in patients]
    return {
        "benchmark_id": "triad5-rashi",
        "created_at": _now(),
        "source_commit": git_commit(),
        "patient_count": len(runs),
        "runs": runs,
    }


def execute_one(patient_id: str, **kwargs) -> dict[str, Any]:
    return execute_patient(get_patient(patient_id), **kwargs)


def slim(record: dict) -> dict:
    """JSON suitable for the dashboard. Raw engine objects stay out of the page."""
    copy = deepcopy(record)
    for tracks_map in copy.get("cells", {}).values():
        for track, cell in list(tracks_map.items()):
            tracks_map[track] = _public_cell(cell)
    return copy
