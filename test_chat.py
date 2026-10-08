import httpx
import time

def test_chat():
    base_url = ""
    session_id = f"test_session_{int(time.time())}"
    patient_id = "002"  # or whichever valid patient exists

    print(f"Starting test for patient {patient_id}")
    
    # Start chat
    start_payload = {
        "session_id": session_id,
        "patient_id": patient_id
    }
    
    print("\n[req] POST /start-consultation", start_payload)
    resp = httpx.post(f"{base_url}/start-consultation", json=start_payload, timeout=30.0)
    print(f"[res] {resp.status_code}")
    print(resp.json())
    
    # Send message 1
    msg1 = "I clearly feel a pain in my back."
    msg_payload_1 = {
        "session_id": session_id,
        "message": msg1
    }
    print(f"\n[req] POST /send-message - {msg1}")
    resp = httpx.post(f"{base_url}/send-message", json=msg_payload_1, timeout=30.0)
    print(f"[res] {resp.status_code}")
    print(resp.json())
    
if __name__ == "__main__":
    test_chat()
