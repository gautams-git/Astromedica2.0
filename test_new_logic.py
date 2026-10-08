import httpx
import json
import asyncio
import time

async def main():
    base_url = ""

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

    print(f"\\n[req] POST /generate-chart")
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{base_url}/generate-chart", json=payload)
        
    try:
        data = resp.json()
        print(f"\\n[res] {resp.status_code}")
        if "disease_filter" in data:
            print("\\n=== DISEASE FILTER OUTPUT ===")
            print(json.dumps(data["disease_filter"], indent=2))
        else:
            print("disease_filter key not found in response!")
            print(json.dumps(data, indent=2))
    except Exception as e:
        print(e)
        print(resp.text)

if __name__ == "__main__":
    asyncio.run(main())
