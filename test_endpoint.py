import json, urllib.request

# Load the real rule1 result
with open('rule1_result.json') as f:
    rule1 = json.load(f)

# Simulate answers (hi-concern)
answers = [
    {"question": "Do you currently experience any issues related to your kidney?",  "answer": "Yes",         "category": "Current Symptoms"},
    {"question": "Do you currently experience any issues related to your ovaries?", "answer": "Yes",         "category": "Current Symptoms"},
    {"question": "Have you noticed any speech related problems recently?",           "answer": "Yes",         "category": "Body Systems"},
    {"question": "Have you or a close family member ever experienced ulcers?",      "answer": "Yes, myself", "category": "Medical History"},
    {"question": "Do you frequently experience any of these: dryness...",           "answer": "Frequently",  "category": "Ayurvedic Constitution"},
    {"question": "How would you rate your mental stress level currently?",          "answer": "Very High",   "category": "Mental Peace"},
    {"question": "How are your overall energy levels through the day?",             "answer": "Very Poor",   "category": "Energy Levels"},
    {"question": "How is your sleep quality on most nights?",                       "answer": "Very Poor",   "category": "Lifestyle"},
]

payload = json.dumps({"rule1": rule1, "answers": answers}).encode()
req = urllib.request.Request(
    "/score-risks",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(req) as r:
    result = json.loads(r.read())

print(f"success: {result['success']}")
print(f"total={result['total']}  high={result['high']}  moderate={result['moderate']}  low={result['low']}")
print()
for r in result['risks'][:6]:
    print(f"  [{r['risk_level']:8}] {r['condition']:<28} score={r['total_score']}")
