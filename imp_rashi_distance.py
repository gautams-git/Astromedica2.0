SIGN_TO_ABSOLUTE_HOUSE = {
    "Aries": 1, "Mesha": 1,
    "Taurus": 2, "Vrishabha": 2, "Rishabha": 2,
    "Gemini": 3, "Mithuna": 3, "Mithun": 3,
    "Cancer": 4, "Karka": 4, "Kataka": 4,
    "Leo": 5, "Simha": 5,
    "Virgo": 6, "Kanya": 6,
    "Libra": 7, "Tula": 7,
    "Scorpio": 8, "Vrischika": 8, "Vrishchika": 8,
    "Sagittarius": 9, "Dhanu": 9, "Dhanur": 9,
    "Capricorn": 10, "Makara": 10,
    "Aquarius": 11, "Kumbha": 11,
    "Pisces": 12, "Meena": 12,
}

INCREASE_SET = {2, 6, 7, 8, 12}
LOWER_SET = {1, 3, 4, 5, 9, 10, 11}


def _active_dasha(dasha_result: dict) -> dict:
    return {
        "mahadasha": ((dasha_result or {}).get("mahadasha") or {}).get("planet", "—"),
        "antardasha": ((dasha_result or {}).get("antardasha") or {}).get("planet", "—"),
        "pratyantardasha": ((dasha_result or {}).get("pratyantardasha") or {}).get("planet", "—"),
    }


def _classify_distance(distance, active_levels):
    bucket = "unmapped"
    effect = "No rule"
    if distance in INCREASE_SET:
        bucket = "increase"
        effect = "Increase Severity" if active_levels else "Increase Pattern Present"
    elif distance in LOWER_SET:
        bucket = "lower"
        effect = "Lower / Nullify" if active_levels else "Lower / Nullify Pattern"
    return bucket, effect


def _role_entries(rule1_result: dict) -> list:
    entries = []

    for index, item in enumerate(rule1_result.get("rogkaraka_1", []) or [], start=1):
        if item and item.get("name"):
            entries.append({
                "role": "P1" if index == 1 else f"P1.{index}",
                "role_base": "P1",
                "role_label": "Occupant",
                "planet": item.get("name"),
                "birth_sign": item.get("sign", ""),
            })

    rk3 = rule1_result.get("rogkaraka_3") or {}
    if rk3.get("name"):
        entries.append({
            "role": "P2",
            "role_base": "P2",
            "role_label": "Dispositor",
            "planet": rk3.get("name"),
            "birth_sign": rk3.get("sign", ""),
        })

    for index, row in enumerate(rule1_result.get("drishti_on_6th", []) or [], start=1):
        pname = row.get("planet", "")
        if pname:
            entries.append({
                "role": "P3" if index == 1 else f"P3.{index}",
                "role_base": "P3",
                "role_label": "Drishti",
                "planet": pname,
                "birth_sign": "",
            })

    rk2 = rule1_result.get("rogkaraka_2") or {}
    if rk2.get("name"):
        entries.append({
            "role": "P4",
            "role_base": "P4",
            "role_label": "6th House Lord",
            "planet": rk2.get("name"),
            "birth_sign": rk2.get("sign", ""),
        })

    return entries


def _active_levels_for_planet(planet_name: str, active_dasha: dict) -> list:
    levels = []
    if planet_name == active_dasha.get("mahadasha"):
        levels.append("Mahadasha")
    if planet_name == active_dasha.get("antardasha"):
        levels.append("Antardasha")
    if planet_name == active_dasha.get("pratyantardasha"):
        levels.append("Pratyantardasha")
    return levels


def build_imp_rashi_distance_analysis(chart_data: dict, rule1_result: dict, current_positions: dict, dasha_result: dict) -> dict:
    planets = chart_data.get("planets", []) or []
    current_planets = (current_positions or {}).get("planets", []) or []
    current_by_name = {p.get("name"): p for p in current_planets if p.get("name")}

    moon = next((p for p in planets if p.get("name") == "Moon"), None)
    if not moon:
        return {
            "available": False,
            "reason": "Moon not found in birth chart planets.",
            "rows": [],
            "trail": {"error": "Moon not found"},
        }

    moon_sign = ((moon.get("sign") or {}).get("name", "—")) if isinstance(moon.get("sign"), dict) else moon.get("sign", "—")
    imp_house = SIGN_TO_ABSOLUTE_HOUSE.get(moon_sign, 0)
    active_dasha = _active_dasha(dasha_result)

    role_rows = []
    for entry in _role_entries(rule1_result):
        planet_name = entry["planet"]
        current_planet = current_by_name.get(planet_name) or {}
        current_abs_house = int(current_planet.get("absolute_house", 0) or 0)
        current_sign = ((current_planet.get("sign") or {}).get("name", "—"))

        raw_gap = ((current_abs_house - imp_house + 12) % 12) if imp_house and current_abs_house else None
        distance = (raw_gap + 1) if raw_gap is not None else None
        active_levels = _active_levels_for_planet(planet_name, active_dasha)
        bucket, effect = _classify_distance(distance, active_levels)

        role_rows.append({
            "role": entry["role"],
            "role_base": entry["role_base"],
            "role_label": entry["role_label"],
            "moon_house": imp_house,
            "planet": planet_name,
            "birth_reference_sign": moon_sign or "—",
            "current_absolute_house": current_abs_house or None,
            "current_sign": current_sign or "—",
            "raw_gap": raw_gap,
            "distance": distance,
            "distance_bucket": bucket,
            "dasha_active_levels": active_levels,
            "effect": effect,
            "note": (
                f"Distance {distance} is in the {'increase' if bucket == 'increase' else 'lower/nullify' if bucket == 'lower' else 'unmapped'} group."
                + (f" Active dasha: {', '.join(active_levels)}." if active_levels else " Planet is not active in current dasha.")
            ) if distance is not None else "Current planetary position not found for this planet.",
            "backend_trail": {
                "reference_house": imp_house,
                "target_absolute_house": current_abs_house or None,
                "raw_gap_formula": f"(({current_abs_house} - {imp_house} + 12) % 12) = {raw_gap}" if current_abs_house else "Target house unavailable",
                "distance_formula": f"{raw_gap} + 1 = {distance}" if raw_gap is not None else "Distance unavailable",
                "increase_set": sorted(INCREASE_SET),
                "lower_set": sorted(LOWER_SET),
                "active_dasha_levels": active_levels,
            }
        })

    return {
        "available": True,
        "title": "IMP Rashi Distance",
        "imp_rashi": {
            "moon_house": imp_house,
            "sign": moon_sign or "—",
            "moon_sign": moon_sign,
        },
        "rules": {
            "increase_distances": sorted(INCREASE_SET),
            "lower_distances": sorted(LOWER_SET),
            "distance_mode": "Astrological house counting (same house = 1)",
            "raw_gap_mode": "Circular gap (same house = 0)",
        },
        "current_dasha": active_dasha,
        "rows": role_rows,
        "trail": {
            "birth_moon": {
                "house": imp_house,
                "sign": moon_sign,
                "degree": moon.get("degree", 0),
                "minutes": moon.get("minutes", 0),
            },
            "imp_rashi_resolution": {
                "house_number_used": imp_house,
                "house_sign_found": moon_sign or "—",
            },
            "current_planetary_timestamp": (current_positions or {}).get("as_of_local", "—"),
            "row_count": len(role_rows),
            "rows": role_rows,
        }
    }


def build_birth_current_distance_analysis(chart_data: dict, rule1_result: dict, current_positions: dict, dasha_result: dict) -> dict:
    planets = chart_data.get("planets", []) or []
    planets_by_name = {p.get("name"): p for p in planets if p.get("name")}
    current_planets = (current_positions or {}).get("planets", []) or []
    current_by_name = {p.get("name"): p for p in current_planets if p.get("name")}
    active_dasha = _active_dasha(dasha_result)

    role_rows = []
    for entry in _role_entries(rule1_result):
        planet_name = entry["planet"]
        birth_planet = planets_by_name.get(planet_name) or {}
        birth_sign = entry.get("birth_sign") or ((birth_planet.get("sign") or {}).get("name", ""))
        birth_abs_house = SIGN_TO_ABSOLUTE_HOUSE.get(birth_sign, 0)

        current_planet = current_by_name.get(planet_name) or {}
        current_abs_house = int(current_planet.get("absolute_house", 0) or 0)
        current_sign = ((current_planet.get("sign") or {}).get("name", "—"))

        raw_gap = ((current_abs_house - birth_abs_house + 12) % 12) if birth_abs_house and current_abs_house else None
        distance = (raw_gap + 1) if raw_gap is not None else None
        active_levels = _active_levels_for_planet(planet_name, active_dasha)
        bucket, effect = _classify_distance(distance, active_levels)

        role_rows.append({
            "role": entry["role"],
            "role_base": entry["role_base"],
            "role_label": entry["role_label"],
            "birth_absolute_house": birth_abs_house or None,
            "birth_sign": birth_sign or "—",
            "planet": planet_name,
            "current_absolute_house": current_abs_house or None,
            "current_sign": current_sign or "—",
            "raw_gap": raw_gap,
            "distance": distance,
            "distance_bucket": bucket,
            "dasha_active_levels": active_levels,
            "effect": effect,
            "note": (
                f"Distance {distance} is in the {'increase' if bucket == 'increase' else 'lower/nullify' if bucket == 'lower' else 'unmapped'} group."
                + (f" Active dasha: {', '.join(active_levels)}." if active_levels else " Planet is not active in current dasha.")
            ) if distance is not None else "Birth or current absolute house could not be resolved for this planet.",
            "backend_trail": {
                "birth_absolute_house": birth_abs_house or None,
                "target_absolute_house": current_abs_house or None,
                "raw_gap_formula": f"(({current_abs_house} - {birth_abs_house} + 12) % 12) = {raw_gap}" if birth_abs_house and current_abs_house else "Birth/current house unavailable",
                "distance_formula": f"{raw_gap} + 1 = {distance}" if raw_gap is not None else "Distance unavailable",
                "increase_set": sorted(INCREASE_SET),
                "lower_set": sorted(LOWER_SET),
                "active_dasha_levels": active_levels,
            }
        })

    return {
        "available": True,
        "title": "Birth vs Current P1-P4 Distance",
        "rules": {
            "increase_distances": sorted(INCREASE_SET),
            "lower_distances": sorted(LOWER_SET),
            "distance_mode": "Astrological house counting (same house = 1)",
            "raw_gap_mode": "Circular gap (same house = 0)",
        },
        "current_dasha": active_dasha,
        "rows": role_rows,
        "trail": {
            "current_planetary_timestamp": (current_positions or {}).get("as_of_local", "—"),
            "row_count": len(role_rows),
            "rows": role_rows,
        }
    }
