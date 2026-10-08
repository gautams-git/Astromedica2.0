
import json
from hitlist_engine import run_hitlist_analysis

# Mock chart data
chart_data = {
    "planets": [
        {
            "name": "Saturn",
            "sign": {"name": "Dhanu"},
            "nakshatra": {"name": "Moola", "pada": 3}
        },
        {
            "name": "Mars",
            "sign": {"name": "Mesha"},
            "nakshatra": {"name": "Ashwini", "pada": 1}
        },
        {
            "name": "Sun",
            "sign": {"name": "Mesha"},
            "nakshatra": {"name": "Ashwini", "pada": 2}
        }
    ],
    "ascendant": {"name": "Mesha"}
}

# 1. Test case: P1 (Occupant) is ABSENT
rule1_result_no_p1 = {
    "rogkaraka_1": [],  # No occupants
    "rogkaraka_2": {"name": "Mars"},  # 6th lord
    "rogkaraka_3": {"name": "Saturn"},  # Dispositor of 6th lord (P2)
    "sixth_house_sign": "Kanya",
    "sixth_house_lord": "Mercury",
    "ascendant_lord": "Mars"
}

print("--- Testing P1 ABSENT ---")
result_no_p1 = run_hitlist_analysis(chart_data, rule1_result_no_p1)
priority_planets = result_no_p1["priority_planets"]

# Verify P2 is at index 0 and has show_rashi_only
p2 = priority_planets[0]
print(f"First planet: {p2['planet']} (Priority: {p2['priority']})")
print(f"Show rashi only: {p2.get('show_rashi_only', False)}")

# Verify P6 (6th House Rashi) is present
p6 = next((p for p in priority_planets if p["priority"] == "P6"), None)
print(f"P6 Rashi: {p6['rashi'] if p6 else 'NOT FOUND'}")

# Verify HL1 and HL2 don't have Saturn (P2) terms
# Saturn's Rashi is Dhanu. Rashi organs for Dhanu should be in HL3.
# Saturn's Planet terms should NOT be in HL1 if show_rashi_only is True.
print(f"HL1 terms count: {len(result_no_p1['hl1'])}")
print(f"HL2 terms count: {len(result_no_p1['hl2'])}")
print(f"HL3 terms count: {len(result_no_p1['hl3'])}")

# 2. Test case: P1 (Occupant) is PRESENT
rule1_result_with_p1 = {
    "rogkaraka_1": [{"name": "Sun"}],  # Occupant
    "rogkaraka_2": {"name": "Mars"},
    "rogkaraka_3": {"name": "Saturn"},
    "sixth_house_sign": "Kanya",
    "sixth_house_lord": "Mercury",
    "ascendant_lord": "Mars"
}

print("\n--- Testing P1 PRESENT ---")
result_with_p1 = run_hitlist_analysis(chart_data, rule1_result_with_p1)
priority_planets_with_p1 = result_with_p1["priority_planets"]

# Verify P1 is at index 0
p1 = priority_planets_with_p1[0]
print(f"First planet: {p1['planet']} (Priority: {p1['priority']})")

# Verify P2 is NOT at index 0 (it should be later in the list)
p2_with_p1 = next((p for p in priority_planets_with_p1 if p["priority"] == "P2"), None)
print(f"P2 position: {priority_planets_with_p1.index(p2_with_p1)}")
print(f"P2 Show rashi only: {p2_with_p1.get('show_rashi_only', False)}")

# Verify P6 is still present
p6_with_p1 = next((p for p in priority_planets_with_p1 if p["priority"] == "P6"), None)
print(f"P6 Rashi: {p6_with_p1['rashi'] if p6_with_p1 else 'NOT FOUND'}")
