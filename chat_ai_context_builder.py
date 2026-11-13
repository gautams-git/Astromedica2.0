#!/usr/bin/env python3
"""
Builds an interview-focused AI context file for a patient.
"""

import json
import os
from datetime import datetime


def _read_json_if_exists(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _ensure_list(value):
    return value if isinstance(value, list) else []


def _trim_list(items, limit=5):
    output = []
    seen = set()
    for item in _ensure_list(items):
        if isinstance(item, dict):
            key = json.dumps(item, sort_keys=True, default=str)
        else:
            key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def _extract_terms_from_trail(trail, prefixes):
    terms = []
    for item in _ensure_list(trail):
        source = str(item.get("source", ""))
        term = str(item.get("term", "")).strip()
        if term and any(prefix in source for prefix in prefixes):
            terms.append(term)
    return _trim_list(terms, limit=6)


def _summarize_hitlist(hitlist: dict) -> dict:
    top4 = _ensure_list(hitlist.get("top4"))[:3]
    top_systems = []
    for cluster in top4:
        top_systems.append({
            "label": cluster.get("label") or cluster.get("display_name") or "Unknown",
            "score": cluster.get("score"),
            "reason": cluster.get("reason") or cluster.get("support_summary") or "",
            "organs": _extract_terms_from_trail(cluster.get("trail", []), ["HL3"]),
            "probable_diseases": _extract_terms_from_trail(cluster.get("trail", []), ["HL1", "HL2"]),
        })
    return {
        "top_systems": top_systems,
        "hitlist_names": _trim_list([item.get("label") for item in top4], limit=3),
    }


def _summarize_disease_list_block(block: dict, top_key: str = "top_diseases") -> dict:
    top_rows = _ensure_list(block.get(top_key))[:5]
    return {
        "top_conditions": [
            {
                "label": row.get("disease") or row.get("condition") or row.get("label") or "Unknown",
                "score": row.get("score") or row.get("total_score") or row.get("final_score"),
                "reason": row.get("support_summary") or row.get("explanation") or "",
            }
            for row in top_rows
        ]
    }


def _summarize_algorithm_comparison(algorithm_comparison: dict) -> list:
    algorithms = algorithm_comparison.get("algorithms", {}) if isinstance(algorithm_comparison, dict) else {}
    summaries = []
    for name, payload in algorithms.items():
        top_items = _ensure_list(payload.get("top_organs_and_systems"))[:3]
        summaries.append({
            "name": name,
            "top_items": [
                {
                    "label": item.get("label") or "Unknown",
                    "score": item.get("score"),
                    "reason": item.get("reason") or item.get("support_summary") or "",
                }
                for item in top_items
            ],
        })
    return summaries


def _build_consensus(processed: dict, diagnosis: dict, final_output: dict) -> dict:
    hitlist_labels = [item.get("label") for item in _ensure_list(processed.get("hitlist", {}).get("top4"))[:3] if item.get("label")]
    rashi_labels = [item.get("label") or item.get("disease") or item.get("condition") for item in _ensure_list(processed.get("rashi_correlation", {}).get("top_diseases"))[:5]]
    organ_labels = [item.get("label") or item.get("disease") or item.get("condition") for item in _ensure_list(processed.get("organ_truth_correlation", {}).get("top_diseases"))[:5]]
    diagnosis_labels = _ensure_list(diagnosis.get("high_risk_conditions"))
    common_findings = processed.get("common_findings", {})

    return {
        "strong_agreements": _trim_list(common_findings.get("top_matches", []) or common_findings.get("common_findings", []), limit=5),
        "moderate_agreements": _trim_list(hitlist_labels + rashi_labels + organ_labels, limit=8),
        "diagnosis_focus": _trim_list(diagnosis_labels, limit=8),
        "conflicts": _trim_list(final_output.get("hitlist_confirmation", []), limit=5) if isinstance(final_output, dict) else [],
    }


def build_chat_ai_context(patient_id: str, folder: str | None = None) -> dict:
    folder = folder or os.path.join("patients", str(patient_id))
    processed = _read_json_if_exists(os.path.join(folder, "processed_logic.json"), {})
    diagnosis = _read_json_if_exists(os.path.join(folder, "diagnosis.json"), {})
    final_output = _read_json_if_exists(os.path.join(folder, "final_output.json"), {})
    algorithm_comparison = _read_json_if_exists(os.path.join(folder, "algorithm_comparison.json"), {})
    profile = _read_json_if_exists(os.path.join(folder, "profile.json"), {})

    dasha = processed.get("dasha", {}) if isinstance(processed, dict) else {}
    context = {
        "patient": {
            "patient_id": str(patient_id),
            "name": profile.get("name", ""),
            "gender": profile.get("gender", processed.get("gender", diagnosis.get("gender", "Unknown"))),
            "dob": profile.get("dob", ""),
            "birth_time": profile.get("birth_time", ""),
            "birth_place": profile.get("birth_place", ""),
        },
        "current_dasha_context": {
            "mahadasha": dasha.get("mahadasha", {}),
            "antardasha": dasha.get("antardasha", {}),
            "pratyantardasha": dasha.get("pratyantardasha", {}),
            "active_dasha_alerts": _trim_list(diagnosis.get("active_dasha_alerts", []), limit=5),
            "dasha_chat_priority": processed.get("dasha_chat_priority", {}),
        },
        "algorithm_outputs": {
            "hitlist": _summarize_hitlist(processed.get("hitlist", {})),
            "rashi_correlation": _summarize_disease_list_block(processed.get("rashi_correlation", {})),
            "organ_truth_correlation": _summarize_disease_list_block(processed.get("organ_truth_correlation", {})),
            "disease_first_logic": _summarize_disease_list_block(processed.get("disease_first_logic", {})),
            "disease_compare_logic": _summarize_disease_list_block(processed.get("disease_compare_logic", {})),
            "algorithm_comparison": _summarize_algorithm_comparison(algorithm_comparison),
        },
        "cross_algorithm_consensus": _build_consensus(processed, diagnosis, final_output),
        "combined_output_summary": {
            "available": bool(final_output),
            "risk_grade": final_output.get("overall_risk", {}),
            "top_conditions": _trim_list(final_output.get("disease_probability", []), limit=5),
            "care_period": final_output.get("care_period", {}),
            "doctor_recommendation": final_output.get("doctor_recommendation", {}),
            "hitlist_confirmation": _trim_list(final_output.get("hitlist_confirmation", []), limit=5),
        },
        "questioning_targets": [
            {
                "target": "High-risk conditions from diagnosis",
                "why": "These are the current strongest diagnosis-side conditions before chat confirmation.",
                "items": _trim_list(diagnosis.get("high_risk_conditions", []), limit=8),
            },
            {
                "target": "Cross-algorithm overlap",
                "why": "These are repeated themes across algorithms and should be confirmed early.",
                "items": _trim_list(processed.get("common_findings", {}).get("top_matches", []), limit=8),
            },
            {
                "target": "Organs and systems at risk",
                "why": "Use these to ask symptom-led questions without exposing internal engine labels.",
                "items": _trim_list(diagnosis.get("organs_at_risk", []), limit=8),
            },
        ],
        "question_plan": {
            "opening_focus": _trim_list(diagnosis.get("organs_at_risk", []), limit=3),
            "secondary_focus": _trim_list(diagnosis.get("body_systems", []), limit=3),
            "red_flag_checks": _trim_list(diagnosis.get("high_risk_conditions", []), limit=5),
        },
        "guardrails": {
            "ask_one_question_at_a_time": True,
            "do_not_mention_astrology": True,
            "use_symptom_led_language": True,
            "resolve_conflicts_before_final_summary": True,
        },
        "generated_at": datetime.now().isoformat(),
    }

    out_file = os.path.join(folder, "chat_ai_context.json")
    with open(out_file, "w") as f:
        json.dump(context, f, indent=2)
    return context
