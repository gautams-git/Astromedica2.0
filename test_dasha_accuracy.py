import json
from datetime import date

def test_dasha_accuracy():
    try:
        with open("chart_data.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ chart_data.json not found! Please generate a chart first.")
        return

    today = date.today()
    print(f"📅 Today's Date for testing: {today}")

    dasha_raw = data.get("dasha_raw", {})
    periods = (
        dasha_raw.get("dasha_periods") or
        dasha_raw.get("mahadasha") or
        []
    )

    if not periods:
        print("❌ No dasha periods found in chart_data.json.")
        return

    print("\n🔍 Scanning Dasha Periods from API payload...")
    for maha in periods:
        m_start_str = str(maha.get("start", ""))[:10]
        m_end_str = str(maha.get("end", ""))[:10]
        
        try:
            m_start = date.fromisoformat(m_start_str)
            m_end = date.fromisoformat(m_end_str)
        except: continue

        if m_start <= today <= m_end:
            planet = maha.get("planet", {}).get("name", maha.get("name", "Unknown"))
            print(f"✅ Found Current MAHADASHA: {planet} ({m_start_str} to {m_end_str})")
            
            # Check Antardasha
            for antar in maha.get("antardasha", []):
                a_start_str = str(antar.get("start", ""))[:10]
                a_end_str = str(antar.get("end", ""))[:10]
                try:
                    a_start = date.fromisoformat(a_start_str)
                    a_end = date.fromisoformat(a_end_str)
                except: continue
                
                if a_start <= today <= a_end:
                    a_planet = antar.get("planet", {}).get("name", antar.get("name", "Unknown"))
                    print(f"✅ Found Current ANTARDASHA: {a_planet} ({a_start_str} to {a_end_str})")

                    # Check Pratyantardasha
                    for pratya in antar.get("pratyantardasha", []):
                        p_start_str = str(pratya.get("start", ""))[:10]
                        p_end_str = str(pratya.get("end", ""))[:10]
                        try:
                            p_start = date.fromisoformat(p_start_str)
                            p_end = date.fromisoformat(p_end_str)
                        except: continue

                        if p_start <= today <= p_end:
                            p_planet = pratya.get("planet", {}).get("name", pratya.get("name", "Unknown"))
                            print(f"✅ Found Current PRATYANTARDASHA: {p_planet} ({p_start_str} to {p_end_str})")
                            break
                    break
            break

if __name__ == "__main__":
    test_dasha_accuracy()
