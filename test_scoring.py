import json
from rules import apply_rule_1, score_health_risks

with open('chart_data.json') as f:
    chart = json.load(f)

rule1 = apply_rule_1(chart)

SCENARIO_A = [
    {"question": "Do you currently experience any issues related to your kidney?",   "answer": "Yes",           "category": "Current Symptoms"},
    {"question": "Do you currently experience any issues related to your ovaries?",  "answer": "Yes",           "category": "Current Symptoms"},
    {"question": "Have you noticed any speech related problems recently?",           "answer": "Yes",           "category": "Body Systems"},
    {"question": "Have you or a close family member ever experienced ulcers?",       "answer": "Yes, myself",   "category": "Medical History"},
    {"question": "Do you frequently experience any of these: dryness...",            "answer": "Frequently",    "category": "Ayurvedic Constitution"},
    {"question": "How would you rate your mental stress level currently?",           "answer": "Very High",     "category": "Mental Peace"},
    {"question": "How are your overall energy levels through the day?",              "answer": "Very Poor",     "category": "Energy Levels"},
    {"question": "How is your sleep quality on most nights?",                        "answer": "Very Poor",     "category": "Lifestyle"},
]

SCENARIO_B = [
    {"question": "Do you currently experience any issues related to your kidney?",   "answer": "No",            "category": "Current Symptoms"},
    {"question": "Do you currently experience any issues related to your ovaries?",  "answer": "No",            "category": "Current Symptoms"},
    {"question": "Have you noticed any speech related problems recently?",           "answer": "No",            "category": "Body Systems"},
    {"question": "Have you or a close family member ever experienced ulcers?",       "answer": "No",            "category": "Medical History"},
    {"question": "Do you frequently experience any of these: dryness...",            "answer": "Never",         "category": "Ayurvedic Constitution"},
    {"question": "How would you rate your mental stress level currently?",           "answer": "Low",           "category": "Mental Peace"},
    {"question": "How are your overall energy levels through the day?",              "answer": "Excellent",     "category": "Energy Levels"},
    {"question": "How is your sleep quality on most nights?",                        "answer": "Very Good",     "category": "Lifestyle"},
]

for label, answers in [("SCENARIO A (High concern)", SCENARIO_A),
                        ("SCENARIO B (Low concern)",  SCENARIO_B)]:
    risks = score_health_risks(rule1, answers)
    high  = sum(1 for r in risks if r["risk_level"] == "HIGH")
    mod   = sum(1 for r in risks if r["risk_level"] == "MODERATE")
    low   = sum(1 for r in risks if r["risk_level"] == "LOW")
    print(f"\n{'='*48}")
    print(f"  {label}")
    print(f"  HIGH={high}  MODERATE={mod}  LOW={low}")
    print(f"{'='*48}")
    for r in risks[:8]:
        print(f"  [{r['risk_level']:8}] {r['condition']:<28} score={r['total_score']}")
