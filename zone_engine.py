# zone_engine.py
# ══════════════════════════════════════════
# ZONE SCORE ENGINE — Independent module
# DO NOT modify any existing files except
# adding 2 lines to server.py
# ══════════════════════════════════════════

# Fixed zone numbers for each SIGN (Standard South Indian Layout)
# Mapping: Aries=1, Taurus=2, Gemini=3, Cancer=1, Leo=2, Virgo=3, 
#          Libra=1, Scorpio=2, Sagittarius=3, Capricorn=1, Aquarius=2, Pisces=3
SIGN_ZONES = {
    "Aries": 1, "Mesha": 1,
    "Taurus": 2, "Vrishabha": 2, "Rishabha": 2,
    "Gemini": 3, "Mithuna": 3, "Mithun": 3,
    "Cancer": 1, "Karka": 1, "Kataka": 1,
    "Leo": 2, "Simha": 2,
    "Virgo": 3, "Kanya": 3,
    "Libra": 1, "Tula": 1,
    "Scorpio": 2, "Vrischika": 2, "Vrishchika": 2,
    "Sagittarius": 3, "Dhanu": 3, "Dhanur": 3,
    "Capricorn": 1, "Makara": 1,
    "Aquarius": 2, "Kumbha": 2,
    "Pisces": 3, "Meena": 3
}

SIGN_LORDS = {
    "Mesha":     "Mars",
    "Aries":     "Mars",
    "Vrishabha": "Venus",
    "Rishabha":  "Venus",
    "Taurus":    "Venus",
    "Mithuna":   "Mercury",
    "Mithun":    "Mercury",
    "Gemini":    "Mercury",
    "Karka":     "Moon",
    "Kataka":    "Moon",
    "Cancer":    "Moon",
    "Simha":     "Sun",
    "Leo":       "Sun",
    "Kanya":     "Mercury",
    "Virgo":     "Mercury",
    "Tula":      "Venus",
    "Libra":     "Venus",
    "Vrischika": "Mars",
    "Vrishchika":"Mars",
    "Scorpio":   "Mars",
    "Dhanu":     "Jupiter",
    "Dhanur":    "Jupiter",
    "Sagittarius":"Jupiter",
    "Makara":    "Saturn",
    "Capricorn": "Saturn",
    "Kumbha":    "Saturn",
    "Aquarius":  "Saturn",
    "Meena":     "Jupiter",
    "Pisces":    "Jupiter",
}


def get_8th_house_sign(houses):
    """
    From houses list find the sign
    of the 8th house.
    """
    for h in houses:
        if h.get("house") == 8:
            return h.get("sign", {}).get("name", "")
    return ""


def get_planet_house(planet_name, planets):
    """
    Find which house a planet is sitting in.
    Returns house number and sign name.
    """
    for p in planets:
        if p.get("name") == planet_name:
            return {
                "house": p.get("house", 0),
                "sign":  p.get("sign", {}).get(
                    "name", ""),
                "planet":  planet_name
            }
    return {"house": 0, "sign": "", 
            "planet": planet_name}


def zones_match(z1, z2, target1, target2):
    """
    Check if {z1,z2} matches {target1,target2}
    Order does not matter.
    """
    # Create sorted tuples for comparison to handle order independence
    pair = tuple(sorted((z1, z2)))
    target = tuple(sorted((target1, target2)))
    return pair == target


def run_zone_analysis(chart_data, second_ascendant=None):
    """
    Main entry point.
    Takes chart_data from server.py
    Returns full zone score result with trail.
    """
    planets = chart_data.get("planets", [])
    houses  = chart_data.get("houses", [])
    ascendant = chart_data.get(
        "ascendant", {})

    trail = []
    second_ascendant = second_ascendant or {}

    # ── Step 1: Find Ascendant sign and lord ──
    asc_sign = ascendant.get("name", "")
    asc_lord = SIGN_LORDS.get(asc_sign, "")

    trail.append({
        "step": "Ascendant Sign",
        "value": asc_sign,
        "note": f"Ascendant is {asc_sign}"
    })
    trail.append({
        "step": "Ascendant Lord",
        "value": asc_lord,
        "note": f"Lord of {asc_sign} is {asc_lord}"
    })

    # ── Step 2: Find where Asc Lord sits ──────
    asc_lord_data = get_planet_house(
        asc_lord, planets)
    asc_lord_house = asc_lord_data["house"]
    asc_lord_sign = asc_lord_data["sign"]
    asc_lord_zone = SIGN_ZONES.get(asc_lord_sign, 0)

    # Backend Trail Format: asc lord (name) - rashi (name) - zone (number)
    trail.append({
        "step": "Asc Lord Analysis",
        "value": f"{asc_lord} - {asc_lord_sign} - Z{asc_lord_zone}",
        "note": f"Asc Lord {asc_lord} sits in {asc_lord_sign} (House {asc_lord_house})"
    })

    # ── Step 3: Find 8th House Lord ────────
    eighth_house_sign = get_8th_house_sign(houses)
    eighth_lord = SIGN_LORDS.get(eighth_house_sign, "")
    eighth_lord_data = get_planet_house(eighth_lord, planets)
    eighth_lord_house = eighth_lord_data["house"]
    eighth_lord_sign = eighth_lord_data["sign"]
    eighth_lord_zone = SIGN_ZONES.get(eighth_lord_sign, 0)

    # Backend Trail Format: 8th lord (name) - rashi (name) - zone (number)
    trail.append({
        "step": "8th Lord Analysis",
        "value": f"{eighth_lord} - {eighth_lord_sign} - Z{eighth_lord_zone}",
        "note": f"8th Lord {eighth_lord} sits in {eighth_lord_sign} (House {eighth_lord_house})"
    })

    # ── Step 4: Find Moon and Saturn zones ───
    moon_data = get_planet_house("Moon", planets)
    moon_house = moon_data["house"]
    moon_sign = moon_data["sign"]
    moon_zone = SIGN_ZONES.get(moon_sign, 0)

    saturn_data = get_planet_house("Saturn", planets)
    saturn_house = saturn_data["house"]
    saturn_sign = saturn_data["sign"]
    saturn_zone = SIGN_ZONES.get(saturn_sign, 0)

    trail.append({
        "step": "Moon Analysis",
        "value": f"Moon - House {moon_house} - Z{moon_zone}",
        "note": f"Moon sits in {moon_sign} (House {moon_house})"
    })
    trail.append({
        "step": "Saturn Analysis",
        "value": f"Saturn - House {saturn_house} - Z{saturn_zone}",
        "note": f"Saturn sits in {saturn_sign} (House {saturn_house})"
    })

    # ── Step 5: Calculate score ──────────────
    asc_zone = SIGN_ZONES.get(asc_sign, 0)
    second_asc_result = second_ascendant.get("result", {})
    asc2_sign = second_asc_result.get("rashi", "")
    asc2_house = second_asc_result.get("house", 0)
    asc2_zone = SIGN_ZONES.get(asc2_sign, 0)

    trail.append({
        "step": "Asc Analysis",
        "value": f"Asc - {asc_sign} - Z{asc_zone}",
        "note": f"Ascendant is in {asc_sign} (Zone {asc_zone})"
    })
    trail.append({
        "step": "2nd Asc Analysis",
        "value": f"Asc2 - {asc2_sign} - Z{asc2_zone}",
        "note": f"2nd Ascendant is in {asc2_sign} (House {asc2_house}, Zone {asc2_zone})"
    })

    array1 = (asc_lord_zone, eighth_lord_zone)
    array2 = (moon_zone, saturn_zone)
    array3 = (asc_zone, asc2_zone)
    
    def get_score(z1, z2):
        if zones_match(z1, z2, 1, 1) or zones_match(z1, z2, 2, 3):
            return 10, "{1,1} or {2,3} → 10 pts"
        if zones_match(z1, z2, 1, 3) or zones_match(z1, z2, 2, 2):
            return 5, "{1,3} or {2,2} → 5 pts"
        if zones_match(z1, z2, 1, 2) or zones_match(z1, z2, 3, 3):
            return 7, "{1,2} or {3,3} → 7 pts"
        return 0, "No match → 0 pts"

    s1, m1 = get_score(array1[0], array1[1])
    s2, m2 = get_score(array2[0], array2[1])
    s3, m3 = get_score(array3[0], array3[1])
    
    raw_score = s1 + s2 + s3
    score = 27 if raw_score == 22 else raw_score
    matched_rule = f"Array1: {m1} | Array2: {m2} | Array3: {m3} | Total: {score}"
    if raw_score != score:
        matched_rule += f" (rounded from {raw_score})"

    trail.append({
        "step": "Zone Score",
        "value": score,
        "note": matched_rule
    })

    # ── Prepare result ──────────────────────
    return {
        "asc_sign": asc_sign,
        "asc_lord": asc_lord,
        "asc_lord_house": asc_lord_house,
        "asc_lord_sign": asc_lord_sign,
        "asc_lord_zone": asc_lord_zone,
        
        "eighth_lord": eighth_lord,
        "eighth_lord_house": eighth_lord_house,
        "eighth_lord_sign": eighth_lord_sign,
        "eighth_lord_zone": eighth_lord_zone,
        
        "moon_house": moon_house,
        "moon_sign": moon_sign,
        "moon_zone": moon_zone,
        
        "saturn_house": saturn_house,
        "saturn_sign": saturn_sign,
        "saturn_zone": saturn_zone,

        "asc_zone": asc_zone,
        "asc2_house": asc2_house,
        "asc2_sign": asc2_sign,
        "asc2_zone": asc2_zone,
        
        "array1_display": f"{{{array1[0]}, {array1[1]}}}",
        "array2_display": f"{{{array2[0]}, {array2[1]}}}",
        "array3_display": f"{{{array3[0]}, {array3[1]}}}",
        "array1_score": s1,
        "array2_score": s2,
        "array3_score": s3,
        
        "raw_score": raw_score,
        "score": score,
        "matched_rule": matched_rule,
        "trail": trail
    }
