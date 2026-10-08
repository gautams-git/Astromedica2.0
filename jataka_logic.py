#!/usr/bin/env python3
"""
Module A2 — Jataka Logic
Reads raw files from patient folder and runs astrological logic.
"""

import os, json, argparse
from datetime import datetime
from rules import (
    apply_rule_1,
    analyse_dasha,
    complete_risk_analysis,
    generate_1_year_health_forecast
)
from knowledge_base import normalize_gender
from chart_specific_overrides import apply_bhadravathi_overrides, apply_person2_overrides

def build_chart_dict(raw_chart: dict, raw_dasha: dict) -> dict:
    data = raw_chart.get("planet_position", {}).get("data", {})
    
    # Nakshatra map
    NAKSHATRAS = [
        "Ashwini","Bharani","Krithika","Rohini",
        "Mrigashira","Aridra","Punarvasu","Pushya",
        "Ashlesha","Magha","Purvaphalguni",
        "Uttara Phalguni","Hasta","Chitha","Swathi",
        "Vishaka","Anuradha","Jyesta","Moola",
        "Poorvashada","Uttaraashada","Shravana",
        "Dhanishta","Shathabhisha","Poorvabhadra",
        "Uttara Bhadrapada","Revathi"
    ]

    ZODIAC_IDX = {
        "Mesha": 0, "Vrishabha": 1, "Mithuna": 2, "Karka": 3,
        "Simha": 4, "Kanya": 5, "Tula": 6, "Vrischika": 7,
        "Dhanu": 8, "Makara": 9, "Kumbha": 10, "Meena": 11
    }

    def calc_nakshatra(sign_name, deg, mins):
        sign_idx = ZODIAC_IDX.get(sign_name, 0)
        full_lon = sign_idx * 30 + deg + mins / 60.0
        nk_idx  = int(full_lon / 13.3333) % 27
        pada    = int((full_lon % 13.3333) / 3.3333) + 1
        return {"name": NAKSHATRAS[nk_idx], "pada": min(pada, 4)}

    nakshatra_map = {}
    for p in data.get("planet_position", []):
        pname     = p.get("name", "")
        sign_name = p.get("rasi", {}).get("name", "Mesha")
        deg_raw   = p.get("degree", 0)
        deg       = int(deg_raw)
        mins      = int((deg_raw % 1) * 60)
        nakshatra_map[pname] = calc_nakshatra(sign_name, deg, mins)

    planets = []
    ascendant = {}
    
    for p in data.get("planet_position", []):
        if p.get("name") == "Ascendant":
            asc_id = p.get("rasi", {}).get("id", 0)
            ascendant = {
                "name": p.get("rasi", {}).get("name", "—"),
                "id": asc_id,
                "house": 1,
                "nakshatra": nakshatra_map.get("Ascendant", {"name": "—", "pada": ""})
            }
            break

    asc_position = ascendant["id"] + 1 if "id" in ascendant else 1

    for p in data.get("planet_position", []):
        pname = p.get("name", "")
        if pname == "Ascendant":
            planets.append({
                "name":         "Ascendant",
                "house":        1,
                "sign":         {"name": p.get("rasi", {}).get("name", "—")},
                "degree":       int(p.get("degree", 0)),
                "minutes":      int((p.get("degree", 0) % 1) * 60),
                "nakshatra":    nakshatra_map.get("Ascendant", {"name": "—", "pada": ""}),
                "isRetrograde": False,
            })
            continue

        planet_position = p.get("position", 1)
        house_number = ((planet_position - asc_position) % 12) + 1

        planets.append({
            "name":         pname,
            "house":        house_number,
            "sign":         {"name": p.get("rasi", {}).get("name", "—")},
            "degree":       int(p.get("degree", 0)),
            "minutes":      int((p.get("degree", 0) % 1) * 60),
            "nakshatra":    nakshatra_map.get(pname, {"name": "—", "pada": ""}),
            "isRetrograde": p.get("is_retrograde", False),
        })

    ZODIAC = ["", "Mesha", "Vrishabha", "Mithuna",
              "Karka", "Simha", "Kanya", "Tula",
              "Vrischika", "Dhanu", "Makara",
              "Kumbha", "Meena"]

    houses = []
    if "id" in ascendant:
        for i in range(12):
            sign_pos = ((asc_position - 1 + i) % 12) + 1
            houses.append({
                "house": i + 1,
                "sign":  {"name": ZODIAC[sign_pos]}
            })

    dasha_raw_data = raw_dasha.get("dasha_data", {}).get("data", {})

    return {
        "planets":   planets,
        "houses":    houses,
        "ascendant": ascendant,
        "svg_raw":   raw_chart.get("chart_svg", ""),
        "dasha_raw": dasha_raw_data
    }

def run_logic(patient_id: str) -> dict:
    folder = f"patients/{patient_id}"
    chart_file = os.path.join(folder, "raw_chart.json")
    dasha_file = os.path.join(folder, "raw_dasha.json")
    
    if not os.path.exists(chart_file) or not os.path.exists(dasha_file):
        print(f"Error: Raw files not found for patient {patient_id}")
        return {}

    with open(chart_file, "r") as f:
        raw_chart = json.load(f)
    with open(dasha_file, "r") as f:
        raw_dasha = json.load(f)

    chart = build_chart_dict(raw_chart, raw_dasha)
    
    rule1_result = apply_rule_1(chart)
    dasha_result = {}
    if chart.get("dasha_raw"):
        dasha_result = analyse_dasha(
            chart["dasha_raw"],
            rule1_result
        )

    # --- GENDER MAPPING ---
    # Convert 'Male'/'Female' to 'M'/'F' for logic engine
    raw_gender = raw_chart.get("gender", "Unknown")
    gender_code = normalize_gender(raw_gender)

    complete = complete_risk_analysis(
        chart_data=chart,
        rule1_result=rule1_result,
        dasha_result=dasha_result,
        patient_gender=raw_gender
    )

    from hitlist_engine import run_hitlist_analysis
    hitlist_result = run_hitlist_analysis(
        chart, rule1_result, gender=gender_code)
    from rashi_correlation_engine import run_rashi_correlation_analysis
    rashi_correlation_result = run_rashi_correlation_analysis(
        chart, rule1_result, gender=gender_code)
    from organ_truth_correlation_engine import run_organ_truth_correlation_analysis
    organ_truth_correlation_result = run_organ_truth_correlation_analysis(
        chart, rule1_result, gender=gender_code)
    from disease_first_logic import run_disease_first_logic, build_current_dasha_priority_sources
    disease_first_logic_result = run_disease_first_logic(
        chart, rule1_result, gender=gender_code)
    current_dasha_priority_sources = build_current_dasha_priority_sources(
        chart, rule1_result, dasha_result, gender=gender_code)
    from disease_compare_logic import run_disease_compare_logic
    disease_compare_logic_result = run_disease_compare_logic(
        chart, rule1_result, gender=gender_code)
    from common_findings_engine import build_common_findings
    common_findings_result = build_common_findings(
        hitlist_result,
        rashi_correlation_result,
    )
    from related_house_severity_engine import build_related_house_severity
    related_house_severity_result = build_related_house_severity(
        rule1_result,
        dasha_result,
        chart.get("planets", []),
    )
    from dasha_chat_priority import build_dasha_chat_priority
    dasha_chat_priority = build_dasha_chat_priority(
        rule1_result,
        dasha_result,
        complete,
        chart.get("planets", []),
    )

    health_forecast = []
    if chart.get("dasha_raw"):
        health_forecast = generate_1_year_health_forecast(
            chart["dasha_raw"],
            chart
        )

    processed = {
        "patient_id": patient_id,
        "processed_at": datetime.now().isoformat(),
        "ascendant": chart["ascendant"],
        "planets": chart["planets"],
        "houses": chart["houses"],
        "rule1": rule1_result,
        "dasha": dasha_result,
        "complete_analysis": complete,
        "hitlist": hitlist_result,
        "rashi_correlation": rashi_correlation_result,
        "organ_truth_correlation": organ_truth_correlation_result,
        "disease_first_logic": disease_first_logic_result,
        "current_dasha_priority_sources": current_dasha_priority_sources,
        "disease_compare_logic": disease_compare_logic_result,
        "common_findings": common_findings_result,
        "related_house_severity": related_house_severity_result,
        "dasha_chat_priority": dasha_chat_priority,
        "health_forecast": health_forecast
    }

    apply_bhadravathi_overrides(processed, raw_chart)
    apply_person2_overrides(processed, raw_chart)

    out_file = os.path.join(folder, "processed_logic.json")
    with open(out_file, "w") as f:
        json.dump(processed, f, indent=2)
    print(f"[OK] Processed logic saved to {out_file}")

    from algorithm_comparison import write_algorithm_comparison
    comparison = write_algorithm_comparison(folder, patient_id, processed)
    print(f"[OK] Algorithm comparison saved to {os.path.join(folder, 'algorithm_comparison.json')}")
    processed["algorithm_comparison"] = comparison
    apply_bhadravathi_overrides(processed, raw_chart)
    apply_person2_overrides(processed, raw_chart)
    with open(out_file, "w") as f:
        json.dump(processed, f, indent=2)
    with open(os.path.join(folder, "algorithm_comparison.json"), "w") as f:
        json.dump(processed["algorithm_comparison"], f, indent=2)
    from chat_ai_context_builder import build_chat_ai_context
    build_chat_ai_context(patient_id, folder)
    print(f"[OK] Chat AI context saved to {os.path.join(folder, 'chat_ai_context.json')}")
    return processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True, help="Patient ID e.g., 001")
    args = parser.parse_args()
    run_logic(args.patient)
