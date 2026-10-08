from knowledge_base import get_nakshatra_diseases, NAKSHATRA_DISEASES
import json

def verify():
    print("--- Verifying Nakshatra Disease Lookup ---")
    
    # Test 1: Ashwini (uniform padas)
    ash1 = get_nakshatra_diseases("Ashwini", 1)
    ash2 = get_nakshatra_diseases("Ashwini", 2)
    print(f"Ashwini P1: {ash1[:2]}...")
    print(f"Ashwini P2: {ash2[:2]}...")
    
    # Test 2: Mrigashira (split padas)
    mri1 = get_nakshatra_diseases("Mrigashira", 1)
    mri3 = get_nakshatra_diseases("Mrigashira", 3)
    print(f"Mrigashira P1: {mri1[:2]}...") # Should be Pimples
    print(f"Mrigashira P3: {mri3[:2]}...") # Should be Corrupted blood
    
    assert "Pimples" in mri1[0]
    assert "Corrupted blood" in mri3[0]
    
    # Test 3: Fallback logic
    fb = get_nakshatra_diseases("Mrigashira", 9) # Invalid pada
    print(f"Mrigashira P9 (fallback): {fb[:2]}...")
    
    print("\n✅ Verification Successful!")

if __name__ == "__main__":
    try:
        verify()
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
