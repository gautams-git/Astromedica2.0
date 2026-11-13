"""Pilot weight search. It does not write production weights.

The twelve-patient set is too small for a generalisable learner. Training uses
only the train split. Holdout ids are refused. The reward is verification and
stability, not an LLM critic vote and not a clinical label.
"""

from __future__ import annotations

import random
from copy import deepcopy

from benchmark.oracles import verify_rashi
from benchmark.patients import splits, get_patient
from benchmark.rashi_weights import baseline_tables, validate_proposal
from rashi_correlation_engine import run_rashi_correlation_analysis
from agentic_rashi_correlation.profiles import weights_context


ARMS = (1.0, 1.1, 1.2)


def _top_label(chart, rule1, gender, weights) -> tuple[str, dict]:
    with weights_context(weights):
        engine = run_rashi_correlation_analysis(deepcopy(chart), deepcopy(rule1), gender=gender)
    groups = engine.get("all_groups") or []
    label = groups[0]["label"] if groups else ""
    verification = verify_rashi(chart, rule1, gender, engine)
    return label, verification


def _reward(chart, rule1, gender, weights, baseline_label: str) -> float:
    label, verification = _top_label(chart, rule1, gender, weights)
    score = 1.0 if verification.get("status") == "passed" else -1.0
    if label != baseline_label:
        score -= 0.25
    return score


def _weights_for(multiplier: float) -> dict:
    baseline = baseline_tables()
    proposal = {
        "layer_weights": {"nakshatra": round(baseline["LAYER_WEIGHTS"]["nakshatra"] * multiplier, 4)},
        "justification": f"pilot arm {multiplier}",
    }
    validation = validate_proposal(proposal, baseline)
    if not validation["valid"]:
        raise RuntimeError(validation["errors"])
    return validation["weights"]


def run_pilot(seed: int = 7) -> dict:
    grouped = splits()
    train_ids = list(grouped["train"])
    holdout = set(grouped["holdout"])
    if not train_ids:
        return {
            "status": "skipped",
            "claim": "No saved patient charts were available. Prokerala was not called, and no charts were invented.",
            "train_ids": [],
            "holdout_ids_excluded": sorted(holdout),
            "validation_ids_excluded": list(grouped["validation"]),
            "selected_for_inspection_only": {"production_weights_written": False},
            "bandit": {"pulls": []},
        }
    if set(train_ids) & holdout:
        raise RuntimeError("train and holdout overlap")

    patients = []
    for patient_id in train_ids:
        patient = get_patient(patient_id)
        if patient_id in holdout:
            raise RuntimeError("holdout leaked into training")
        patients.append(patient)

    baseline = baseline_tables()
    baselines = {}
    for patient in patients:
        chart, rule1, gender = patient["chart"], patient["rule1"], patient["chart"].get("gender")
        label, _verification = _top_label(chart, rule1, gender, baseline)
        baselines[patient["patient_id"]] = label

    def mean_reward(multiplier: float) -> float:
        weights = _weights_for(multiplier)
        total = 0.0
        for patient in patients:
            chart, rule1 = patient["chart"], patient["rule1"]
            gender = chart.get("gender")
            total += _reward(chart, rule1, gender, weights, baselines[patient["patient_id"]])
        return total / len(patients)

    grid = [{"arm": arm, "mean_reward": mean_reward(arm)} for arm in ARMS]
    rng = random.Random(seed)
    random_arms = [rng.choice(ARMS) for _ in range(4)]
    random_search = [{"arm": arm, "mean_reward": mean_reward(arm)} for arm in random_arms]

    # Epsilon-greedy over the same three arms. Context is the patient case name,
    # but the arm value is shared: this is a pilot, not a fitted policy.
    counts = {arm: 0 for arm in ARMS}
    values = {arm: 0.0 for arm in ARMS}
    epsilon = 0.25
    pulls = []
    for step, patient in enumerate(patients):
        if rng.random() < epsilon:
            arm = rng.choice(ARMS)
            explore = True
        else:
            arm = max(ARMS, key=lambda item: values[item])
            explore = False
        weights = _weights_for(arm)
        chart, rule1 = patient["chart"], patient["rule1"]
        reward = _reward(chart, rule1, chart.get("gender"), weights, baselines[patient["patient_id"]])
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        pulls.append({"step": step, "patient_id": patient["patient_id"], "arm": arm, "explore": explore, "reward": reward})

    best_grid = max(grid, key=lambda item: item["mean_reward"])
    best_bandit = max(ARMS, key=lambda item: values[item])
    return {
        "status": "pilot",
        "claim": "Pipeline demonstration on six synthetic training charts. Not evidence of generalisable or clinical improvement.",
        "train_ids": train_ids,
        "holdout_ids_excluded": sorted(holdout),
        "validation_ids_excluded": list(grouped["validation"]),
        "reward": "verification pass minus a penalty when the top label moves away from the default-weight label",
        "grid_search": grid,
        "random_search": random_search,
        "bandit": {"values": values, "counts": counts, "pulls": pulls, "selected_arm": best_bandit},
        "selected_for_inspection_only": {
            "grid_arm": best_grid["arm"],
            "bandit_arm": best_bandit,
            "production_weights_written": False,
        },
    }
