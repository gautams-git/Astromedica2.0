
import json
from chat_module import build_system_prompt

def test_hitlist_prompt():
    # Mock data
    rule1_result = {"rogkaraka_1": [], "rogkaraka_2": {}, "rogkaraka_3": {}}
    complete_analysis = {"organ_disease_map": [], "disease_risks": []}
    dasha_result = {"alerts": []}
    hitlist_result = {
        "top4": [
            {
                "label": "Hormonal & Metabolic",
                "score": 45.5,
                "convergence": 3,
                "trail": [
                    {"source": "HL3/Rashi", "term": "Pancreas"},
                    {"source": "HL2/Nakshatra", "term": "Diabetes"},
                    {"source": "HL1/Planet", "term": "Insulin resistance"}
                ]
            },
            {
                "label": "Digestive / Liver",
                "score": 38.2,
                "convergence": 2,
                "trail": [
                    {"source": "HL3/Rashi", "term": "Liver"},
                    {"source": "HL2/Nakshatra", "term": "Fatty liver"}
                ]
            },
            {
                "label": "Nervous System",
                "score": 30.1,
                "convergence": 1,
                "trail": [
                    {"source": "HL3/Rashi", "term": "Nerves"},
                    {"source": "HL2/Nakshatra", "term": "Anxiety"}
                ]
            },
            {
                "label": "Musculoskeletal",
                "score": 25.4,
                "convergence": 1,
                "trail": [
                    {"source": "HL3/Rashi", "term": "Bones"},
                    {"source": "HL2/Nakshatra", "term": "Arthritis"}
                ]
            }
        ]
    }
    
    prompt = build_system_prompt(
        rule1_result,
        complete_analysis,
        dasha_result,
        gender="Male",
        hitlist_result=hitlist_result
    )
    
    print("--- GENERATED PROMPT PREVIEW ---")
    # print(prompt.encode('utf-8', errors='replace').decode('utf-8'))
    
    # Check if HitList data is in prompt
    assert "Hormonal & Metabolic" in prompt
    assert "Pancreas" in prompt
    assert "Diabetes" in prompt
    assert "Liver" in prompt
    assert "Fatty liver" in prompt
    assert "TOP 4 PRIORITY SYSTEMS" in prompt
    assert "GROUP ANALYSIS & PATTERN RECOGNITION" in prompt
    assert "Theme 4 (ask fourth): Musculoskeletal" in prompt
    
    print("\n[SUCCESS] HitList data correctly integrated into system prompt.")

if __name__ == "__main__":
    test_hitlist_prompt()
