#!/usr/bin/env python3
"""
Module A3 — Disease Diagnose
Reads processed logic and maps to diseases.
"""

import os, json, argparse
from datetime import datetime

def run_diagnose(patient_id: str) -> dict:
    folder = f"patients/{patient_id}"
    logic_file = os.path.join(folder, "processed_logic.json")
    disease_file = "data/disease_table.json"
    
    if not os.path.exists(logic_file):
        print(f"Error: {logic_file} not found")
        return {}
    if not os.path.exists(disease_file):
        print(f"Error: {disease_file} not found")
        return {}

    with open(logic_file, "r") as f:
        logic = json.load(f)
    with open(disease_file, "r") as f:
        diseases = json.load(f)

    rule1 = logic.get("rule1", {})
    complete = logic.get("complete_analysis", {})
    dasha = logic.get("dasha", {})
    
    rk_planets = []
    if "rogkaraka_1" in rule1:
        rk_planets.extend(rule1["rogkaraka_1"])
    if "rogkaraka_2" in rule1 and rule1["rogkaraka_2"]:
        rk_planets.append(rule1["rogkaraka_2"])
        
    risk_planet_names = [p["name"] for p in rk_planets]
    
    body_systems = []
    organs_at_risk = []
    dosha_imbalance = []
    dosha_symptoms = {}
    nakshatra_tendencies = []
    
    # Process RK planets
    for p in rk_planets:
        pname = p.get("name", "")
        # Sign info is not uniformly on the RK planet object in rule1 output, 
        # so we extract from chart planets. Wait, is sign there? The prompt says map to organs (RASHI_ORGANS for their sign).
        # We can look up the planet from logic["planets"]
        
        # Let's find the planet in chart["planets"]
        chart_planet = next((cp for cp in logic.get("planets", []) if cp["name"] == pname), {})
        sign = chart_planet.get("sign", {}).get("name", "")
        nk = p.get("nakshatra", "—")
        
        if pname in diseases.get("PLANET_DISEASES", {}):
            d = diseases["PLANET_DISEASES"][pname]
            if d not in body_systems:
                body_systems.append(d)
                
        if sign in diseases.get("RASHI_ORGANS", {}):
            o = diseases["RASHI_ORGANS"][sign]
            if o not in organs_at_risk:
                organs_at_risk.append(o)
                
        if pname in diseases.get("PLANET_DOSHA", {}):
            dosha = diseases["PLANET_DOSHA"][pname]
            if dosha not in dosha_imbalance:
                dosha_imbalance.append(dosha)
                
        if nk in diseases.get("NAKSHATRA_DISEASES", {}):
            nk_data = diseases["NAKSHATRA_DISEASES"][nk]
            pada = str(p.get("nakshatra_pada", "1"))
            
            tend = []
            if isinstance(nk_data, dict):
                tend = nk_data.get(pada) or nk_data.get("1") or []
            elif isinstance(nk_data, list):
                tend = nk_data
                
            for t in tend:
                if t not in nakshatra_tendencies:
                    nakshatra_tendencies.append(t)

    for dosha in dosha_imbalance:
        if dosha in diseases.get("DOSHA_SYMPTOMS", {}):
            dosha_symptoms[dosha] = diseases["DOSHA_SYMPTOMS"][dosha]

    high_risk_conditions = []
    for risk in complete.get("disease_risks", []):
        if risk.get("risk_level") in ["HIGH", "CRITICAL"]:
            if risk["condition"] not in high_risk_conditions:
                high_risk_conditions.append(risk["condition"])
                
    active_dasha_alerts = dasha.get("alerts", [])

    diagnosis = {
        "patient_id": patient_id,
        "diagnosed_at": datetime.now().isoformat(),
        "risk_planets": risk_planet_names,
        "body_systems": body_systems,
        "organs_at_risk": organs_at_risk,
        "dosha_imbalance": dosha_imbalance,
        "dosha_symptoms": dosha_symptoms,
        "nakshatra_tendencies": nakshatra_tendencies,
        "active_dasha_alerts": active_dasha_alerts,
        "high_risk_conditions": high_risk_conditions,
        "risk_score": {
            "step_a": complete.get("step_a_score", 0),
            "step_b": complete.get("step_b_score", 0),
            "step_c": complete.get("step_c_score", 0),
            "total": complete.get("total_score", 0),
            "level": complete.get("overall_risk", "UNKNOWN")
        }
    }

    out_file = os.path.join(folder, "diagnosis.json")
    with open(out_file, "w") as f:
        json.dump(diagnosis, f, indent=2)
    print(f"✅ Diagnosis saved to {out_file}")
    return diagnosis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True, help="Patient ID e.g., 001")
    args = parser.parse_args()
    run_diagnose(args.patient)
