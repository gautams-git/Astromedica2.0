from __future__ import annotations

from typing import Any


TARGET_HOUSES = (2, 8, 11, 12)


def _planet_details_by_name(planets: list[dict] | None) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for planet in planets or []:
        name = str(planet.get("name", "")).strip()
        if name:
            result[name] = planet
    return result


def _active_dasha_levels(dasha: dict | None) -> dict[str, list[str]]:
    dasha = dasha or {}
    level_map: dict[str, list[str]] = {}
    for level_key, label in (
        ("mahadasha", "Mahadasha"),
        ("antardasha", "Antardasha"),
        ("pratyantardasha", "Pratyantardasha"),
    ):
        planet_name = str(((dasha.get(level_key) or {}).get("planet", ""))).strip()
        if not planet_name:
            continue
        level_map.setdefault(planet_name, []).append(label)
    return level_map


def build_related_house_severity(rule1: dict | None, dasha: dict | None, planets: list[dict] | None) -> dict:
    rule1 = rule1 or {}
    relation_map = (rule1.get("house_occupant_relations") or {}) if isinstance(rule1, dict) else {}
    planet_map = _planet_details_by_name(planets)
    dasha_levels = _active_dasha_levels(dasha)

    items: list[dict[str, Any]] = []
    for house_number in TARGET_HOUSES:
        for rel in relation_map.get(house_number, []) or []:
            if not rel.get("related"):
                continue
            name = str(rel.get("name", "")).strip()
            active_levels = dasha_levels.get(name, [])
            if not name or not active_levels:
                continue

            planet = planet_map.get(name, {})
            items.append({
                "planet": name,
                "house": int(rel.get("house", house_number) or house_number),
                "connected_role": rel.get("connected_role", ""),
                "connection_reason": rel.get("connection_reason", ""),
                "active_dasha_levels": active_levels,
                "severity_label": "Increased Severity",
                "message": (
                    f"{name} is related to Rule 1 through {str(rel.get('connected_role', '')).replace('_', ' ')} "
                    f"and is currently active in {', '.join(active_levels)}."
                ),
                "sign": (
                    ((planet.get("sign") or {}).get("name", ""))
                    if isinstance(planet.get("sign"), dict)
                    else planet.get("sign", "")
                ),
                "degree": planet.get("degree"),
                "minutes": planet.get("minutes"),
                "nakshatra": (
                    ((planet.get("nakshatra") or {}).get("name", ""))
                    if isinstance(planet.get("nakshatra"), dict)
                    else planet.get("nakshatra", "")
                ),
            })

    items.sort(key=lambda item: (len(item["active_dasha_levels"]), -item["house"]), reverse=True)

    return {
        "title": "Related House Severity Increase",
        "subtitle": "2nd, 8th, 11th and 12th house planets that are Rule 1 related and active in current dasha.",
        "items": items,
        "count": len(items),
    }
