import json

with open('chart_data.json') as f:
    d = json.load(f)

print("=" * 50)
print("DASHA RAW STRUCTURE CHECK")
print("=" * 50)

raw = d.get('dasha_raw', {})

if not raw:
    print("❌ dasha_raw is EMPTY")
    print("   Either Dasha API call failed")
    print("   OR chart was generated before")
    print("   the Dasha API call was added")
    print()
    print("   Fix: Regenerate chart through")
    print("   browser at ")
else:
    print(f"✅ dasha_raw found")
    print(f"   Top level keys: {list(raw.keys())}")
    print()

    periods = (
        raw.get('dasha_periods') or
        raw.get('mahadasha') or
        raw.get('dashas') or
        raw.get('vimshottari_dasha') or
        []
    )

    print(f"   Periods found: {len(periods)}")
    print()

    if periods:
        print("FIRST MAHADASHA:")
        print(json.dumps(periods[0], indent=2)[:800])
        print()

        antardashas = (
            periods[0].get('antardasha') or
            periods[0].get('sub_periods') or
            periods[0].get('bhukti') or
            []
        )

        if antardashas:
            print(f"   Antardasha count: {len(antardashas)}")
            print()
            print("FIRST ANTARDASHA:")
            print(json.dumps(antardashas[0], indent=2))
            print()

            pratyas = (
                antardashas[0].get('pratyantardasha') or
                antardashas[0].get('sub_periods') or
                antardashas[0].get('antardasha') or
                []
            )

            if pratyas:
                print(f"   Pratyantardasha count: {len(pratyas)}")
                print()
                print("FIRST PRATYANTARDASHA:")
                print(json.dumps(pratyas[0], indent=2))
            else:
                print("❌ No Pratyantardasha found")
                print(f"   Antardasha keys: {list(antardashas[0].keys())}")
        else:
            print("❌ No Antardasha found")
            print(f"   Period keys: {list(periods[0].keys())}")
    else:
        print("❌ No periods found")
        print(f"   Raw structure:")
        print(json.dumps(raw, indent=2)[:500])

print("=" * 50)
