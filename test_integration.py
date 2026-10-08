import json
from jataka_logic import run_logic

def test_integration():
    patient_id = "002"
    print(f"Running logic for patient_id: {patient_id}")
    try:
        result = run_logic(patient_id)
        
        # Check if hitlist exists in result
        if "hitlist" in result:
            print("[SUCCESS] HitList generated successfully.")
            hitlist = result["hitlist"]
            
            # Print top 4 systems
            print("\nTop 4 Priority Systems:")
            top4 = hitlist.get("top4", [])
            for sys in top4:
                label = sys.get("label", sys.get("display_name", "Unknown"))
                score = sys.get("score", 0)
                print(f"- {label}: {score:.2f}")
                
            # Check if drishti_on_6th is present in rule1
            rule1 = result.get("rule1", {})
            if "drishti_on_6th" in rule1:
                print(f"\n[SUCCESS] drishti_on_6th found in rule1: {rule1['drishti_on_6th']}")
            else:
                print("\n[FAILURE] drishti_on_6th NOT found in rule1.")
                
        else:
            print("[FAILURE] HitList NOT found in result.")
            
    except Exception as e:
        print(f"[ERROR] Error during logic execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration()
