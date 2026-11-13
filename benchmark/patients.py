"""Saved-patient loader.

The benchmark reads patient folders that already contain Prokerala results
(`raw_chart.json` and `raw_dasha.json`). It rebuilds the chart with the local
`build_chart_dict` / `apply_rule_1` functions. It does not call Prokerala and
it does not invent a chart when those files are missing.

Set BENCHMARK_PATIENTS_DIR to point at another folder of the same shape.
The default is the application's `patients/` directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def patients_root() -> Path:
    configured = os.environ.get("BENCHMARK_PATIENTS_DIR", "").strip()
    if configured:
        return Path(configured)
    return ROOT / "patients"


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def saved_patient_ids() -> list[str]:
    root = patients_root()
    if not root.is_dir():
        return []
    found = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "raw_chart.json").is_file() and (entry / "raw_dasha.json").is_file():
            found.append(entry.name)
    return found[:12]


def _split_for(index: int) -> str:
    if index < 6:
        return "train"
    if index < 8:
        return "validation"
    return "holdout"


def load_saved_patient(patient_id: str) -> dict[str, Any]:
    """Load one saved chart. Raises FileNotFoundError if Prokerala files are absent.

    The Prokerala client is intentionally not imported.
    """
    folder = patients_root() / patient_id
    chart_path = folder / "raw_chart.json"
    dasha_path = folder / "raw_dasha.json"
    if not chart_path.is_file() or not dasha_path.is_file():
        raise FileNotFoundError(
            f"Patient {patient_id} has no saved Prokerala files at {folder}. "
            "The benchmark will not call Prokerala to create them."
        )
    raw_chart = _read(chart_path)
    raw_dasha = _read(dasha_path)
    profile = _read(folder / "profile.json") if (folder / "profile.json").is_file() else {}

    from jataka_logic import build_chart_dict
    from knowledge_base import normalize_gender
    from rules import apply_rule_1

    chart = build_chart_dict(raw_chart, raw_dasha)
    gender = normalize_gender(raw_chart.get("gender") or profile.get("gender"))
    chart["gender"] = gender
    chart["dasha"] = chart.get("dasha_raw") or {}
    rule1 = apply_rule_1(chart)
    ids = saved_patient_ids()
    index = ids.index(patient_id) if patient_id in ids else 0
    return {
        "patient_id": patient_id,
        "pseudonym": patient_id,
        "split": _split_for(index),
        "case": "saved_prokerala_chart",
        "note": "Chart and dasha were read from saved files. Prokerala was not called.",
        "profile": {
            "name": profile.get("name") or raw_chart.get("name") or f"Patient {patient_id}",
            "dob": profile.get("dob") or raw_chart.get("dob") or "",
            "birth_place": profile.get("birth_place") or raw_chart.get("place") or raw_chart.get("birth_place") or "",
            "gender": gender or "",
        },
        "chart": chart,
        "rule1": rule1,
        "source": {
            "raw_chart": str(chart_path),
            "raw_dasha": str(dasha_path),
            "prokerala_called": False,
        },
        "reference": None,
        "reference_note": "No adjudicated clinical label is attached to saved charts.",
    }


def all_patients() -> list[dict[str, Any]]:
    return [load_saved_patient(patient_id) for patient_id in saved_patient_ids()]


def get_patient(patient_id: str) -> dict[str, Any]:
    folder = patients_root() / patient_id
    if not (folder / "raw_chart.json").is_file() or not (folder / "raw_dasha.json").is_file():
        raise FileNotFoundError(
            f"Patient {patient_id} has no saved Prokerala files. The benchmark will not call Prokerala."
        )
    return load_saved_patient(patient_id)


def is_benchmark_patient(patient_id: str) -> bool:
    return patient_id in saved_patient_ids()


def index_rows() -> list[dict[str, Any]]:
    rows = []
    for patient in all_patients():
        profile = patient["profile"]
        rows.append({
            "patient_id": patient["patient_id"],
            "name": profile["name"],
            "dob": profile["dob"],
            "birth_place": profile["birth_place"],
            "gender": profile["gender"],
            "benchmark": True,
            "case": patient["case"],
            "split": patient["split"],
            "prokerala_called": False,
        })
    return rows


def splits() -> dict[str, list[str]]:
    grouped = {"train": [], "validation": [], "holdout": []}
    for patient_id in saved_patient_ids():
        index = saved_patient_ids().index(patient_id)
        grouped[_split_for(index)].append(patient_id)
    return grouped
