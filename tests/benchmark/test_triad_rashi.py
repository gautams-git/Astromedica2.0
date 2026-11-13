"""Benchmark tests against saved Prokerala files.

The twelve charts are written as local raw_chart.json / raw_dasha.json files.
Nothing in this module is allowed to call Prokerala.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("BENCHMARK_RASHI_WEIGHT_MODE", "mock")
os.environ.setdefault("LLM_PROVIDER", "stub")
os.environ.pop("GEMINI_API_KEY", None)

from benchmark.agent_eval import evaluate_scenarios
from benchmark.architecture import ALGORITHM_MAP
from benchmark.controller import execute_one
from benchmark.metrics import hit_rate_at_k, mean_reciprocal_rank, precision_at_k, rank_correlation, recall_at_k
from benchmark.optimize import run_pilot
from benchmark.patients import all_patients, get_patient, saved_patient_ids, splits
from benchmark.rashi_weights import mock_proposal, propose_weights, resolve_mode, validate_proposal
from benchmark.reports import executive_report


def _write_saved_chart(root, patient_id: str, sign_id: int) -> None:
    folder = root / patient_id
    folder.mkdir(parents=True)
    signs = [
        (0, "Mesha"), (1, "Vrishabha"), (2, "Mithuna"), (3, "Karka"),
        (4, "Simha"), (5, "Kanya"), (6, "Tula"), (7, "Vrischika"),
        (8, "Dhanu"), (9, "Makara"), (10, "Kumbha"), (11, "Meena"),
    ]
    asc = signs[sign_id % 12]
    planets = [
        {"name": "Ascendant", "rasi": {"name": asc[1], "id": asc[0]}, "degree": 5.0, "position": 1},
        {"name": "Sun", "rasi": {"name": "Simha", "id": 4}, "degree": 10.0, "position": 5},
        {"name": "Mars", "rasi": {"name": "Karka", "id": 3}, "degree": 12.5, "position": 4},
        {"name": "Mercury", "rasi": {"name": "Mithuna", "id": 2}, "degree": 8.0, "position": 3},
        {"name": "Venus", "rasi": {"name": "Vrishabha", "id": 1}, "degree": 18.0, "position": 2},
        {"name": "Saturn", "rasi": {"name": "Makara", "id": 9}, "degree": 3.0, "position": 10},
        {"name": "Jupiter", "rasi": {"name": "Dhanu", "id": 8}, "degree": 21.0, "position": 9},
        {"name": "Moon", "rasi": {"name": "Karka", "id": 3}, "degree": 2.0, "position": 4},
        {"name": "Rahu", "rasi": {"name": "Meena", "id": 11}, "degree": 14.0, "position": 12},
        {"name": "Ketu", "rasi": {"name": "Kanya", "id": 5}, "degree": 14.0, "position": 6},
    ]
    (folder / "raw_chart.json").write_text(json.dumps({
        "name": f"Saved {patient_id}",
        "dob": "1990-01-15",
        "gender": "Male" if sign_id % 2 == 0 else "Female",
        "place": "Saved chart",
        "planet_position": {"data": {"planet_position": planets}},
    }), encoding="utf-8")
    (folder / "raw_dasha.json").write_text(json.dumps({"dasha_data": {"data": {}}}), encoding="utf-8")
    (folder / "profile.json").write_text(json.dumps({
        "name": f"Saved {patient_id}",
        "dob": "1990-01-15",
        "birth_place": "Saved chart",
        "gender": "Male" if sign_id % 2 == 0 else "Female",
    }), encoding="utf-8")


@pytest.fixture
def saved_twelve(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PATIENTS_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHMARK_RASHI_WEIGHT_MODE", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for index in range(12):
        _write_saved_chart(tmp_path, f"{101 + index}", index)
    return tmp_path


def test_architecture_limits_track_c_to_rashi():
    enabled = [row["algorithm"] for row in ALGORITHM_MAP if row["track_c"]]
    assert enabled == ["Rashi Correlation"]


def test_twelve_saved_charts_are_loaded_without_prokerala(saved_twelve, monkeypatch):
    import jataka_fetch

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Prokerala was called")

    monkeypatch.setattr(jataka_fetch, "fetch_and_save", _forbidden)
    ids = saved_patient_ids()
    assert ids == [str(101 + index) for index in range(12)]
    patients = all_patients()
    assert len(patients) == 12
    assert all(item["source"]["prokerala_called"] is False for item in patients)
    assert all(item["chart"]["planets"] for item in patients)
    assert all(item["rule1"] for item in patients)
    grouped = splits()
    assert set(grouped["train"]) & set(grouped["holdout"]) == set()
    assert len(grouped["train"]) == 6


def test_missing_saved_chart_does_not_call_prokerala(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_PATIENTS_DIR", str(tmp_path))
    import jataka_fetch
    monkeypatch.setattr(jataka_fetch, "fetch_and_save", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Prokerala")))
    assert saved_patient_ids() == []
    with pytest.raises(FileNotFoundError):
        get_patient("001")


def test_rashi_tracks_on_saved_chart_match_and_trace_weights(saved_twelve):
    record = execute_one("101", algorithms=("Rashi Correlation",))
    cells = record["cells"]["Rashi Correlation"]
    assert cells["A"]["verification"]["status"] == "passed"
    assert cells["B"]["verification"]["status"] == "passed"
    assert record["differentials"]["Rashi Correlation"]["A_vs_B"]["status"] == "equivalent"
    assert cells["C"]["ok"] is True
    assert cells["C"]["weight_audit"]["live_model"] is False
    assert cells["C"]["verification"]["status"] == "passed"
    assert record["differentials"]["Rashi Correlation"]["A_vs_C"]["status"] == "explained_weight_change"
    hashes = {cell["input_hash"] for cell in cells.values()}
    assert len(hashes) == 1


def test_all_twelve_saved_charts_run_rashi(saved_twelve):
    failures = []
    for patient_id in saved_patient_ids():
        record = execute_one(patient_id, algorithms=("Rashi Correlation",))
        for track, cell in record["cells"]["Rashi Correlation"].items():
            if not cell["ok"]:
                failures.append((patient_id, track, cell["error"]))
            elif cell["verification"]["status"] not in {"passed", "limited"}:
                failures.append((patient_id, track, cell["verification"]))
    assert failures == []


def test_other_algorithms_do_not_get_a_fabricated_track_c(saved_twelve):
    record = execute_one("102", algorithms=("HitList", "Rashi Correlation"), tracks=("A", "C"))
    assert record["cells"]["HitList"]["C"]["ok"] is False
    assert "not implemented" in record["cells"]["HitList"]["C"]["error"]
    assert record["cells"]["Rashi Correlation"]["C"]["ok"] is True


def test_vertex_login_selects_live_without_an_api_key(monkeypatch):
    monkeypatch.delenv("BENCHMARK_RASHI_WEIGHT_MODE", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("benchmark.rashi_weights._vertex_ready", lambda: True)
    assert resolve_mode() == "live"
    monkeypatch.setattr(
        "benchmark.rashi_weights._call_vertex",
        lambda _prompt: (
            '{"layer_weights": {"nakshatra": 1.5}, "anchor_weights": {"P6": 3.6}, "justification": "vertex"}',
            "vertex:gemini-2.5-flash",
            0.0,
        ),
    )
    record = propose_weights({"gender": "Male"})
    assert record["live_model"] is True
    assert record["transport"] == "vertex_adc"
    assert record["validation"]["valid"] is True
    assert record["validation"]["changed"]


def test_missing_vertex_login_stays_on_the_scripted_selector(monkeypatch):
    monkeypatch.delenv("BENCHMARK_RASHI_WEIGHT_MODE", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("benchmark.rashi_weights._vertex_ready", lambda: False)
    assert resolve_mode() == "mock"


def test_invalid_weights_rejected():
    rejected = validate_proposal({"layer_weights": {"nakshatra": 99}, "new_table": {}})
    assert rejected["valid"] is False
    assert rejected["weights"] is None
    assert validate_proposal(mock_proposal())["valid"] is True


def test_scripted_track_c_repeats(saved_twelve):
    tops = []
    for _ in range(5):
        cell = execute_one("103", algorithms=("Rashi Correlation",), tracks=("C",))["cells"]["Rashi Correlation"]["C"]
        assert cell["ok"] is True
        tops.append(tuple((item["label"], item["score"]) for item in cell["rankings"]))
    assert len(set(tops)) == 1


def test_critic_and_metrics_stay_independent_of_an_llm():
    report = evaluate_scenarios()
    assert report["counts"]["fp"] == 0
    assert report["counts"]["fn"] == 0
    ranking = ["Head", "Stomach", "Kidney", "Skin"]
    truth = {"Stomach"}
    assert hit_rate_at_k(ranking, truth, 1) == 0.0
    assert precision_at_k(ranking, truth, 4) == 0.25
    assert recall_at_k(ranking, truth, 2) == 1.0
    assert mean_reciprocal_rank(ranking, truth) == 0.5
    assert rank_correlation(ranking, list(reversed(ranking))) < 0


def test_pilot_uses_only_the_train_split(saved_twelve):
    pilot = run_pilot()
    used = set(pilot["train_ids"]) | {item["patient_id"] for item in pilot["bandit"]["pulls"]}
    assert used.isdisjoint(pilot["holdout_ids_excluded"])
    assert pilot["selected_for_inspection_only"]["production_weights_written"] is False


def test_report_names_the_saved_patient(saved_twelve):
    record = execute_one("112", algorithms=("Rashi Correlation",))
    text = executive_report({"patient_count": 1, "runs": [record]})
    assert "112" in text
    assert "Prokerala" not in text or "clinical" in text


def test_index_lists_saved_charts_and_does_not_call_prokerala(saved_twelve, monkeypatch):
    import jataka_fetch
    monkeypatch.setattr(jataka_fetch, "fetch_and_save", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Prokerala")))
    import server
    patient = get_patient("101")
    monkeypatch.setattr(server, "_prepare_patient_chart", lambda _patient_id: None)
    monkeypatch.setattr(server, "_build_latest_patient_payload", lambda _patient_id: {
        "name": patient["profile"]["name"],
        "gender": patient["chart"].get("gender"),
        "planets": patient["chart"]["planets"],
        "houses": patient["chart"]["houses"],
        "ascendant": patient["chart"]["ascendant"],
        "rule1": patient["rule1"],
        "dasha": patient["chart"].get("dasha") or {},
    })
    client = server.app.test_client()
    listed = client.get("/patients-list")
    assert listed.status_code == 200
    body = listed.get_json()
    assert body["prokerala_called"] is False
    ids = [item["patient_id"] for item in body["patients"]]
    assert ids == [str(101 + index) for index in range(12)]

    catalog = client.get("/benchmark/api/catalog")
    assert catalog.status_code == 200
    assert catalog.get_json()["prokerala_called"] is False
    assert len(catalog.get_json()["patients"]) == 12

    opened = client.get("/patient-data/101")
    assert opened.status_code == 200
    payload = opened.get_json()
    assert payload["prokerala_called"] is False
    tracks = payload["data"]["rashi_tracks"]["tracks"]
    assert tracks["A"]["verification"]["status"] == "passed"
    assert tracks["C"]["ok"] is True
    assert payload["data"]["rashi_tracks"]["prokerala_called"] is False
