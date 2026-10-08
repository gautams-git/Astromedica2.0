from __future__ import annotations

from knowledge_base import NAKSHATRA_DISEASES, PLANET_DISEASES, RASHI_ORGANS, is_term_allowed_for_gender, normalize_gender
from rashi_truth_review import ALL_ORGANS, NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS


SOURCE_WEIGHTS = {
    "P1": 6.0,
    "P2": 4.5,
    "P3": 3.5,
    "P4": 2.8,
    "PM": 2.5,
    "P5": 2.0,
}

PLANET_LAYER_WEIGHTS = {
    "P1": 3.0,
    "P2": 2.2,
    "P3": 1.5,
    "P4": 1.2,
    "PM": 1.5,
    "P5": 1.2,
}

RASHI_LAYER_WEIGHTS = {
    "P1_RASHI": 3.0,
    "P2_RASHI": 2.0,
    "P6": 3.0,
    "PM_RASHI": 1.5,
    "P7": 1.5,
}

SKIP_PREFIXES = ("Further everything", "Further all")
ALL_ORGANS_LOOKUP = {str(item).strip().lower(): item for item in ALL_ORGANS}


def _norm(value: str) -> str:
    return str(value or "").strip().lower()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = _norm(item)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _term_allowed(term: str, gender: str | None) -> bool:
    return is_term_allowed_for_gender(term, gender)


def _planet_terms(planet_name: str) -> list[str]:
    raw = PLANET_DISEASES.get(planet_name, "")
    terms: list[str] = []
    for term in str(raw).split(","):
        clean = term.strip()
        if not clean or any(clean.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        terms.append(clean)
    return terms


def _nakshatra_terms(nakshatra_name: str, pada: int | str) -> list[str]:
    nk_data = NAKSHATRA_DISEASES.get(nakshatra_name, {})
    if not nk_data:
        return []
    pada_str = str(pada)
    for key, diseases in nk_data.items():
        padas_in_key = [p.strip() for p in str(key).split(",")]
        if pada_str in padas_in_key:
            return list(diseases)
    return []


def _rashi_terms(rashi_name: str) -> list[str]:
    raw = RASHI_ORGANS.get(rashi_name, "")
    organs: list[str] = []
    for term in str(raw).split(","):
        clean = term.strip()
        canonical = ALL_ORGANS_LOOKUP.get(clean.lower())
        if canonical:
            organs.append(canonical)
    return _dedupe(organs)


def _planet_mapped_organs(planet_name: str, gender: str | None) -> list[str]:
    organs: list[str] = []
    for term in _planet_terms(planet_name):
        canonical = ALL_ORGANS_LOOKUP.get(term.lower())
        if canonical and _term_allowed(canonical, gender):
            organs.append(canonical)
    return _dedupe(organs)


def _planet_data(chart_data: dict, planet_name: str) -> dict:
    return next((planet for planet in chart_data.get("planets", []) if planet.get("name") == planet_name), {}) or {}


def _build_source(priority: str, label: str, planet_data: dict, gender: str | None, include_rashi: bool = False) -> dict | None:
    planet_name = planet_data.get("name", "")
    if not planet_name:
        return None

    sign_name = ((planet_data.get("sign") or {}).get("name", "")) if isinstance(planet_data.get("sign"), dict) else planet_data.get("sign", "")
    nk = planet_data.get("nakshatra", {}) or {}
    nk_name = nk.get("name", "") if isinstance(nk, dict) else ""
    nk_pada = nk.get("pada", 1) if isinstance(nk, dict) else planet_data.get("nakshatra_pada", 1)

    diseases = [d for d in _nakshatra_terms(nk_name, nk_pada) if _term_allowed(d, gender)]
    planet_organs = [o for o in _planet_mapped_organs(planet_name, gender) if _term_allowed(o, gender)]
    rashi_organs = [o for o in _rashi_terms(sign_name) if _term_allowed(o, gender)] if include_rashi else []

    return {
        "priority": priority,
        "label": label,
        "planet": planet_name,
        "sign": sign_name,
        "nakshatra": nk_name,
        "pada": nk_pada,
        "diseases": _dedupe(diseases),
        "planet_organs": _dedupe(planet_organs),
        "rashi_organs": _dedupe(rashi_organs),
    }


def _new_disease_entry(disease: str) -> dict:
    return {
        "disease": disease,
        "score": 0.0,
        "source_priorities": set(),
        "source_labels": set(),
        "supporting_organs": {},
        "planet_support": {},
        "rashi_support": {},
    }


def _add_organ_support(entry: dict, organ: str, score: float, source_key: str, label: str) -> None:
    organ_entry = entry["supporting_organs"].setdefault(organ, {
        "organ": organ,
        "score": 0.0,
        "supports": [],
    })
    organ_entry["score"] += score
    organ_entry["supports"].append({
        "source": source_key,
        "label": label,
        "score": round(score, 2),
    })


def run_disease_first_logic(chart_data: dict, rule1_result: dict, gender: str | None = None) -> dict:
    gender = normalize_gender(gender or chart_data.get("gender"))
    sources: list[dict] = []
    planets_by_name = {planet.get("name"): planet for planet in chart_data.get("planets", [])}

    occupant_list = rule1_result.get("rogkaraka_1", []) or []
    has_occupant = bool(occupant_list)

    for rk in occupant_list:
        source = _build_source("P1", "Occupant of 6th House", planets_by_name.get(rk.get("name", ""), {}), gender, include_rashi=True)
        if source:
            sources.append(source)

    if not has_occupant:
        for rk in rule1_result.get("eleventh_house_occupants", []) or []:
            source = _build_source("P1", "11th House Occupant as P1", planets_by_name.get(rk.get("name", ""), {}), gender, include_rashi=False)
            if source:
                source["fallback_reason"] = "6th house occupant absent; 11th house occupant becomes first priority."
                sources.append(source)

    rk3 = rule1_result.get("rogkaraka_3") or {}
    source = _build_source("P2", "Dispositor of 6th Lord", planets_by_name.get(rk3.get("name", ""), {}), gender, include_rashi=True)
    if source:
        sources.append(source)

    for drishti in rule1_result.get("drishti_on_6th", []) or []:
        source = _build_source("P3", "Drishti on 6th House", planets_by_name.get(drishti.get("planet", ""), {}), gender, include_rashi=True)
        if source:
            sources.append(source)

    source = _build_source("P4", "6th House Lord", planets_by_name.get(rule1_result.get("sixth_house_lord", ""), {}), gender, include_rashi=True)
    if source:
        sources.append(source)

    source = _build_source("P5", "Ascendant Lord", planets_by_name.get(rule1_result.get("ascendant_lord", ""), {}), gender, include_rashi=True)
    if source:
        sources.append(source)

    conditional_moon = rule1_result.get("conditional_moon") or {}
    if conditional_moon.get("include"):
        source = _build_source("PM", "Conditional Moon", _planet_data(chart_data, "Moon"), gender, include_rashi=True)
        if source:
            sources.append(source)

    standalones: list[dict] = []
    standalones.append({
        "priority": "P7",
        "label": "Ascendant Rashi",
        "organs": [o for o in _rashi_terms(((chart_data.get("ascendant") or {}).get("name", ""))) if _term_allowed(o, gender)],
    })
    if conditional_moon.get("include"):
        moon_sign = (((_planet_data(chart_data, "Moon").get("sign") or {}).get("name", "")))
        standalones.append({
            "priority": "PM_RASHI",
            "label": "Moon Rashi",
            "organs": [o for o in _rashi_terms(moon_sign) if _term_allowed(o, gender)],
        })

    disease_scores: dict[str, dict] = {}

    for source in sources:
        for disease in source.get("diseases", []):
            disease_organs = [o for o in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []) if _term_allowed(o, gender)]
            if not disease_organs:
                continue

            entry = disease_scores.setdefault(disease, _new_disease_entry(disease))
            base_weight = SOURCE_WEIGHTS.get(source["priority"], 1.0)
            entry["score"] += base_weight
            entry["source_priorities"].add(source["priority"])
            entry["source_labels"].add(source["label"])

            matched_planet_organs = [o for o in source.get("planet_organs", []) if o in disease_organs]
            if matched_planet_organs:
                support_entry = entry["planet_support"].setdefault(source["planet"], {
                    "planet": source["planet"],
                    "priority": source["priority"],
                    "matched_organs": set(),
                })
                support_entry["matched_organs"].update(matched_planet_organs)
                for organ in matched_planet_organs:
                    organ_weight = PLANET_LAYER_WEIGHTS.get(source["priority"], 1.0)
                    entry["score"] += organ_weight
                    _add_organ_support(entry, organ, organ_weight, source["priority"], source["label"])

            for organ in [o for o in source.get("rashi_organs", []) if o in disease_organs]:
                rashi_key = f"{source['priority']}_RASHI"
                rashi_weight = RASHI_LAYER_WEIGHTS.get(rashi_key, 1.0)
                entry["score"] += rashi_weight
                tag = entry["rashi_support"].setdefault(rashi_key, {
                    "priority": rashi_key,
                    "label": f"{source['label']} Rashi",
                    "matched_organs": set(),
                })
                tag["matched_organs"].add(organ)
                _add_organ_support(entry, organ, rashi_weight, rashi_key, f"{source['label']} Rashi")

    for standalone in standalones:
        priority = standalone["priority"]
        organs = standalone.get("organs", [])
        if not organs:
            continue
        for disease, entry in disease_scores.items():
            disease_organs = [o for o in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []) if _term_allowed(o, gender)]
            matched_organs = [o for o in organs if o in disease_organs]
            if not matched_organs:
                continue
            tag = entry["rashi_support"].setdefault(priority, {
                "priority": priority,
                "label": standalone["label"],
                "matched_organs": set(),
            })
            weight = RASHI_LAYER_WEIGHTS.get(priority, 1.0)
            for organ in matched_organs:
                tag["matched_organs"].add(organ)
                entry["score"] += weight
                _add_organ_support(entry, organ, weight, priority, standalone["label"])

    from eleventh_lord_confirmation import (
        confirmation_boost,
        find_disease_confirmation,
        get_eleventh_lord_confirmation_source,
    )
    confirmation_source = get_eleventh_lord_confirmation_source(
        chart_data, rule1_result, gender
    )
    confirmations = []
    for disease, entry in disease_scores.items():
        match = find_disease_confirmation(disease, confirmation_source)
        if not match:
            continue
        boost = confirmation_boost(entry["score"])
        if boost <= 0:
            continue
        entry["score"] += boost
        entry["eleventh_lord_confirmation"] = {**match, "boost": round(boost, 2)}
        confirmations.append({"disease": disease, **match, "boost": round(boost, 2)})

    ranked = []
    for disease, entry in disease_scores.items():
        supporting_organs = []
        for organ_data in entry["supporting_organs"].values():
            supports = sorted(organ_data["supports"], key=lambda item: item["score"], reverse=True)
            supporting_organs.append({
                "organ": organ_data["organ"],
                "score": round(organ_data["score"], 2),
                "supports": supports,
            })
        supporting_organs.sort(key=lambda item: item["score"], reverse=True)

        planet_support = []
        for support in entry["planet_support"].values():
            planet_support.append({
                "planet": support["planet"],
                "priority": support["priority"],
                "matched_organs": sorted(support["matched_organs"]),
            })
        planet_support.sort(key=lambda item: ("P1 P2 P3 P4 PM P5".split().index(item["priority"]) if item["priority"] in "P1 P2 P3 P4 PM P5".split() else 99, item["planet"]))

        rashi_support = []
        order = ["P1_RASHI", "P2_RASHI", "P6", "PM_RASHI", "P7"]
        for support in entry["rashi_support"].values():
            rashi_support.append({
                "priority": support["priority"],
                "label": support["label"],
                "matched_organs": sorted(support["matched_organs"]),
            })
        rashi_support.sort(key=lambda item: (order.index(item["priority"]) if item["priority"] in order else 99, item["label"]))

        ranked.append({
            "disease": disease,
            "score": round(entry["score"], 2),
            "source_priorities": sorted(entry["source_priorities"]),
            "source_labels": sorted(entry["source_labels"]),
            "supporting_organs": supporting_organs,
            "planet_support": planet_support,
            "rashi_support": rashi_support,
            "eleventh_lord_confirmation": entry.get("eleventh_lord_confirmation"),
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)

    return {
        "title": "Disease First Logic",
        "subtitle": "Disease-first approach using nakshatra diseases with planet-organ and rashi-organ support.",
        "top_diseases": ranked[:8],
        "all_diseases": ranked,
        "eleventh_lord_nakshatra_confirmation": {
            "source": confirmation_source,
            "matches": confirmations,
        },
        "sources_used": [source["priority"] for source in sources],
        "used_p6_fallback": not has_occupant,
        "fallback_note": "11th house occupant/rashi used as first priority because 6th house occupant is absent." if not has_occupant else "",
    }


def build_current_dasha_priority_sources(chart_data: dict, rule1_result: dict, dasha_result: dict, gender: str | None = None) -> dict:
    gender = normalize_gender(gender or chart_data.get("gender"))
    sources: list[dict] = []
    planets_by_name = {planet.get("name"): planet for planet in chart_data.get("planets", [])}

    occupant_list = rule1_result.get("rogkaraka_1", []) or []
    for rk in occupant_list:
        source = _build_source("P1", "Occupant of 6th House", planets_by_name.get(rk.get("name", ""), {}), gender, include_rashi=True)
        if source:
            sources.append(source)

    rk3 = rule1_result.get("rogkaraka_3") or {}
    source = _build_source("P2", "Dispositor of 6th Lord", planets_by_name.get(rk3.get("name", ""), {}), gender, include_rashi=True)
    if source:
        sources.append(source)

    for drishti in rule1_result.get("drishti_on_6th", []) or []:
        source = _build_source("P3", "Drishti on 6th House", planets_by_name.get(drishti.get("planet", ""), {}), gender, include_rashi=True)
        if source:
            sources.append(source)

    source = _build_source("P4", "6th House Lord", planets_by_name.get(rule1_result.get("sixth_house_lord", ""), {}), gender, include_rashi=True)
    if source:
        sources.append(source)

    source = _build_source("P5", "Ascendant Lord", planets_by_name.get(rule1_result.get("ascendant_lord", ""), {}), gender, include_rashi=True)
    if source:
        sources.append(source)

    current_dasha_planets = {
        "Mahadasha": ((dasha_result or {}).get("mahadasha") or {}).get("planet", ""),
        "Antardasha": ((dasha_result or {}).get("antardasha") or {}).get("planet", ""),
        "Pratyantardasha": ((dasha_result or {}).get("pratyantardasha") or {}).get("planet", ""),
    }

    active_sources = []
    seen_keys: set[tuple[str, str]] = set()
    for source in sources:
        planet_name = source.get("planet", "")
        if not planet_name:
            continue
        active_levels = [level for level, active_planet in current_dasha_planets.items() if active_planet == planet_name]
        if not active_levels:
            continue
        dedupe_key = (source.get("priority", ""), planet_name)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        active_sources.append({
            "priority": source.get("priority", ""),
            "label": source.get("label", ""),
            "planet": planet_name,
            "sign": source.get("sign", ""),
            "nakshatra": source.get("nakshatra", ""),
            "pada": source.get("pada", ""),
            "active_levels": active_levels,
            "planet_organs": source.get("planet_organs", []),
            "diseases": source.get("diseases", []),
            "rashi_organs": source.get("rashi_organs", []),
        })

    active_sources.sort(
        key=lambda item: (
            "P1 P2 P3 P4 P5".split().index(item["priority"]) if item["priority"] in "P1 P2 P3 P4 P5".split() else 99,
            item["planet"],
        )
    )

    return {
        "title": "Current Dasha Priority Activation",
        "subtitle": "Shows which P1 to P5 source planets are active now in Mahadasha, Antardasha, or Pratyantardasha.",
        "active_sources": active_sources,
        "has_active_sources": bool(active_sources),
    }
