"""HTTP surface for the saved-patient benchmark.

Routes read patient folders that already contain Prokerala results. They do
not call Prokerala.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from flask import jsonify, send_from_directory

from benchmark.agent_eval import evaluate_scenarios
from benchmark.architecture import ALGORITHM_MAP
from benchmark.controller import execute_one, slim
from benchmark.patients import index_rows, is_benchmark_patient, patients_root, saved_patient_ids, splits
from benchmark.reports import executive_report, technical_report

_LOCK = threading.Lock()
_CACHE: dict[str, dict] = {}
_STATIC = Path(__file__).resolve().parent / "static"


def cached_patient(patient_id: str) -> dict:
    with _LOCK:
        cached = _CACHE.get(f"{patients_root()}::{patient_id}")
        if cached is None:
            cached = slim(execute_one(patient_id, algorithms=("Rashi Correlation",)))
            _CACHE[f"{patients_root()}::{patient_id}"] = cached
        return cached


def _tracks_from_record(record: dict) -> dict:
    rashi = record["cells"]["Rashi Correlation"]
    audit = (rashi.get("C") or {}).get("weight_audit") or {}
    return {
        "patient_id": record["patient_id"],
        "case": record["case"],
        "input_hash": record["input"]["input_hash"],
        "prokerala_called": False,
        "live_model": bool(audit.get("live_model")),
        "model_identifier": audit.get("model_identifier"),
        "prompt_version": audit.get("prompt_version"),
        "note": (
            "Saved Prokerala chart. Track C used the scripted weight selector because Vertex AI login was not available. "
            "Prokerala was not called."
            if audit.get("model_identifier") == "benchmark-mock-rashi-weights-v1"
            else "Saved Prokerala chart. Prokerala was not called."
        ),
        "tracks": {
            track: {
                "ok": (rashi.get(track) or {}).get("ok"),
                "error": (rashi.get(track) or {}).get("error"),
                "top": ((rashi.get(track) or {}).get("rankings") or [])[:5],
                "verification": (rashi.get(track) or {}).get("verification"),
                "confidence": (rashi.get(track) or {}).get("confidence"),
                "fallback_note": (rashi.get(track) or {}).get("fallback_note"),
                "changed_weights": ((rashi.get(track) or {}).get("weight_audit") or {}).get("validation", {}).get("changed"),
                "ablation": ((rashi.get(track) or {}).get("weight_audit") or {}).get("ablation"),
                "agent": (rashi.get(track) or {}).get("agent"),
            }
            for track in ("A", "B", "C")
        },
        "differentials": record["differentials"].get("Rashi Correlation"),
    }


def attach_rashi_tracks(patient_id: str, payload: dict) -> dict:
    """Add the Rashi audit panel to a payload built from saved files."""
    if not is_benchmark_patient(patient_id):
        payload["rashi_tracks"] = {
            "prokerala_called": False,
            "error": "This patient has no saved raw_chart.json and raw_dasha.json. Prokerala was not called.",
        }
        return payload
    payload["rashi_tracks"] = _tracks_from_record(cached_patient(patient_id))
    return payload


def _bundle() -> dict:
    runs = [cached_patient(patient_id) for patient_id in saved_patient_ids()]
    return {
        "runs": runs,
        "patient_count": len(runs),
        "source_commit": runs[0]["source_commit"] if runs else "",
        "created_at": runs[0]["created_at"] if runs else "",
        "prokerala_called": False,
    }


def register_benchmark_routes(app) -> None:
    @app.get("/benchmark")
    def benchmark_dashboard():
        return send_from_directory(_STATIC, "index.html")

    @app.get("/benchmark/api/catalog")
    def benchmark_catalog():
        return jsonify({
            "success": True,
            "algorithms": list(ALGORITHM_MAP),
            "patients": index_rows(),
            "splits": splits(),
            "track_c": "Rashi Correlation only",
            "prokerala_called": False,
            "note": "Patients are the saved folders that already contain Prokerala results. This route does not call Prokerala.",
        })

    @app.get("/benchmark/api/patient/<patient_id>")
    def benchmark_patient(patient_id: str):
        if not is_benchmark_patient(patient_id):
            return jsonify({
                "success": False,
                "error": "No saved Prokerala chart for that id. Prokerala was not called.",
            }), 404
        return jsonify({"success": True, "run": cached_patient(patient_id), "prokerala_called": False})

    @app.get("/benchmark/api/summary")
    def benchmark_summary():
        bundle = _bundle()
        if not bundle["runs"]:
            return jsonify({
                "success": True,
                "executive": "No saved patient folders with raw_chart.json and raw_dasha.json were found. Prokerala was not called.",
                "technical": "",
                "critic": evaluate_scenarios(),
                "patients": [],
                "prokerala_called": False,
            })
        return jsonify({
            "success": True,
            "executive": executive_report(bundle),
            "technical": technical_report(bundle),
            "critic": evaluate_scenarios(),
            "prokerala_called": False,
            "patients": [
                {
                    "patient_id": run["patient_id"],
                    "case": run["case"],
                    "split": run["split"],
                    "input_hash": run["input"]["input_hash"],
                    "rashi": {
                        track: {
                            "ok": (run["cells"]["Rashi Correlation"][track] or {}).get("ok"),
                            "top": ((run["cells"]["Rashi Correlation"][track] or {}).get("rankings") or [])[:3],
                            "verification": ((run["cells"]["Rashi Correlation"][track] or {}).get("verification") or {}).get("status"),
                            "error": (run["cells"]["Rashi Correlation"][track] or {}).get("error"),
                        }
                        for track in ("A", "B", "C")
                    },
                    "differentials": run["differentials"].get("Rashi Correlation"),
                }
                for run in bundle["runs"]
            ],
        })

    @app.get("/benchmark/api/reports")
    def benchmark_reports():
        bundle = _bundle()
        return app.response_class(
            json.dumps({
                "executive": executive_report(bundle) if bundle["runs"] else "No saved charts. Prokerala was not called.",
                "technical": technical_report(bundle) if bundle["runs"] else "",
                "prokerala_called": False,
            }),
            mimetype="application/json",
        )
