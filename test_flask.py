import sys
from server import app
import json

payload = {
    "name": "Anurag Sir Test",
    "dob": "1988-05-04",
    "birth_time": "11:04",
    "lat": 28.704060,
    "lng": 77.102493,
    "birth_place": "Delhi",
    "father_name": "",
    "mother_name": ""
}

print("\\n[req] POST /generate-chart")
with app.test_client() as client:
    resp = client.post('/generate-chart', json=payload)
    print("Status:", resp.status_code)
    try:
        data = resp.json
        if data and "disease_filter" in data:
            print("\\n=== DISEASE FILTER OUTPUT ===")
            print(json.dumps(data["disease_filter"], indent=2))
        else:
            print("No disease_filter found or error occurred.")
            print("Response:", data)
    except Exception as e:
        print("Error:", e)
