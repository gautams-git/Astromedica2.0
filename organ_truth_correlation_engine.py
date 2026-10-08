from __future__ import annotations

from knowledge_base import NAKSHATRA_DISEASES, PLANET_DISEASES, RASHI_ORGANS, is_term_allowed_for_gender, normalize_gender
from rashi_truth_review import (
    ALL_ORGANS,
    NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS,
)


ANCHOR_WEIGHTS = {
    "P6": 6.0,
    "P7": 3.0,
    "PM_RASHI":2,
}

SOURCE_WEIGHTS = {
    "P1": 6.0,
    "P2": 4.5,
    "P3": 3.2,
    "P4": 3.0,
    "P5": 2.0,
    "PM": 3.0,
}

LAYER_WEIGHTS = {
    "planet": 1.0,
    "nakshatra": 2.0,
    "source_rashi": 1.0,
}

P2_ABSENT_OCCUPANT_RASHI_MULTIPLIER = 1.5
SKIP_PREFIXES = ("Further everything", "Further all")


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


ALL_ORGANS_LOOKUP = {str(item).strip().lower(): item for item in ALL_ORGANS}


def _term_allowed(term: str, gender: str | None) -> bool:
    return is_term_allowed_for_gender(term, gender)


def _planet_terms(planet_name: str) -> list[str]:
    raw = PLANET_DISEASES.get(planet_name, "")
    terms: list[str] = []
    for term in str(raw).split(","):
        clean = term.strip()
        if not clean:
            continue
        if any(clean.startswith(prefix) for prefix in SKIP_PREFIXES):
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


def _planet_mapped_organs(planet_name: str, gender: str | None) -> tuple[list[str], list[str]]:
    terms = [t for t in _planet_terms(planet_name) if _term_allowed(t, gender)]
    organs: list[str] = []
    for term in terms:
        canonical = ALL_ORGANS_LOOKUP.get(term.lower())
        if canonical and _term_allowed(canonical, gender):
            organs.append(canonical)
    return _dedupe(terms), _dedupe(organs)


def _nakshatra_mapped_organs(nakshatra_name: str, pada: int | str, gender: str | None) -> tuple[list[str], list[str]]:
    diseases = [d for d in _nakshatra_terms(nakshatra_name, pada) if _term_allowed(d, gender)]
    organs: list[str] = []
    for disease in diseases:
        organs.extend([o for o in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []) if _term_allowed(o, gender)])
    return _dedupe(diseases), _dedupe(organs)


def _planet_data(chart_data: dict, planet_name: str) -> dict:
    return next((planet for planet in chart_data.get("planets", []) if planet.get("name") == planet_name), {}) or {}


def _build_anchor(priority: str, label: str, sign: str, gender: str | None) -> dict:
    organs = [term for term in _rashi_terms(sign) if _term_allowed(term, gender)]
    return {
        "priority": priority,
        "label": label,
        "sign": sign,
        "organs": _dedupe(organs),
    }


def _build_source(priority: str, label: str, planet_data: dict, gender: str | None, include_rashi: bool = False) -> dict | None:
    planet_name = planet_data.get("name", "")
    if not planet_name:
        return None

    sign_name = ((planet_data.get("sign") or {}).get("name", "")) if isinstance(planet_data.get("sign"), dict) else planet_data.get("sign", "")
    nk = planet_data.get("nakshatra", {}) or {}
    nk_name = nk.get("name", "") if isinstance(nk, dict) else ""
    nk_pada = nk.get("pada", 1) if isinstance(nk, dict) else planet_data.get("nakshatra_pada", 1)

    planet_terms, planet_organs = _planet_mapped_organs(planet_name, gender)
    nak_diseases, nak_organs = _nakshatra_mapped_organs(nk_name, nk_pada, gender)
    rashi_organs = [term for term in _rashi_terms(sign_name) if _term_allowed(term, gender)] if include_rashi else []

    return {
        "priority": priority,
        "label": label,
        "planet": planet_name,
        "sign": sign_name,
        "nakshatra": nk_name,
        "pada": nk_pada,
        "planet_terms": planet_terms,
        "planet_organs": planet_organs,
        "nakshatra_diseases": nak_diseases,
        "nakshatra_organs": nak_organs,
        "rashi_organs": _dedupe(rashi_organs),
    }


def _new_organ_entry(organ: str) -> dict:
    return {
        "organ": organ,
        "score": 0.0,
        "anchor_hits": set(),
        "exact_sources": set(),
        "disease_sources": set(),
        "related_diseases": {},
        "trail": [],
    }


def _should_keep_ranked_organ(item: dict) -> bool:
    anchor_hits = set(item.get("anchor_hits", []))
    if not item.get("related_diseases"):
        return False
    # Hide organs that are only supported by Ascendant Rashi.
    if anchor_hits == {"P7"}:
        return False
    return True


def _add_disease_to_organ(entry: dict, disease: str, weight: float, source: dict) -> None:
    disease_entry = entry["related_diseases"].setdefault(disease, {
        "disease": disease,
        "score": 0.0,
        "source_priorities": set(),
        "source_labels": set(),
        "planet_support": {},
    })
    disease_entry["score"] += weight
    disease_entry["source_priorities"].add(source["priority"])
    disease_entry["source_labels"].add(source["label"])
    entry["disease_sources"].add(source["priority"])


def run_organ_truth_correlation_analysis(chart_data: dict, rule1_result: dict, gender: str | None = None) -> dict:
    gender = normalize_gender(gender or chart_data.get("gender"))
    sixth_sign = rule1_result.get("sixth_house_sign", "")
    asc_sign = ((chart_data.get("ascendant") or {}).get("name", "")) or ""

    conditional_moon = rule1_result.get("conditional_moon") or {}
    include_moon = bool(conditional_moon.get("include"))
    moon_planet_data = _planet_data(chart_data, "Moon")
    moon_sign = ((moon_planet_data.get("sign") or {}).get("name", "")) if include_moon else ""

    anchors = [
        _build_anchor("P6", "6th House Rashi", sixth_sign, gender),
        _build_anchor("P7", "Ascendant Rashi", asc_sign, gender),
    ]
    if include_moon and moon_sign:
        anchors.append(_build_anchor("PM_RASHI", "Moon Rashi", moon_sign, gender))

    anchors = [anchor for anchor in anchors if anchor.get("organs")]

    anchor_organs: set[str] = set()
    organ_scores: dict[str, dict] = {}
    for anchor in anchors:
        base_weight = ANCHOR_WEIGHTS.get(anchor["priority"], 1.0)
        for organ in anchor["organs"]:
            anchor_organs.add(organ)
            entry = organ_scores.setdefault(organ, _new_organ_entry(organ))
            entry["score"] += base_weight
            entry["anchor_hits"].add(anchor["priority"])
            entry["trail"].append({
                "source": anchor["priority"],
                "label": anchor["label"],
                "layer": "anchor",
                "weight": round(base_weight, 3),
            })

    sources: list[dict] = []
    planets_by_name = {planet.get("name"): planet for planet in chart_data.get("planets", [])}

    occupant_list = rule1_result.get("rogkaraka_1", []) or []
    has_occupant = bool(occupant_list)

    for rk in occupant_list:
        source = _build_source("P1", "Occupant of 6th House", planets_by_name.get(rk.get("name", ""), {}), gender, include_rashi=False)
        if source:
            sources.append(source)

    if not has_occupant:
        for rk in rule1_result.get("eleventh_house_occupants", []) or []:
            source = _build_source("P1", "11th House Occupant as P1", planets_by_name.get(rk.get("name", ""), {}), gender, include_rashi=False)
            if source:
                source["fallback_reason"] = "6th house occupant absent; 11th house occupant becomes first priority."
                sources.append(source)

    rk3 = rule1_result.get("rogkaraka_3") or {}
    source = _build_source("P2", "Dispositor of 6th Lord", planets_by_name.get(rk3.get("name", ""), {}), gender, include_rashi=False)
    if source:
        sources.append(source)

    for drishti in rule1_result.get("drishti_on_6th", []) or []:
        source = _build_source("P3", "Drishti on 6th House", planets_by_name.get(drishti.get("planet", ""), {}), gender, include_rashi=False)
        if source:
            sources.append(source)

    source = _build_source("P4", "6th House Lord", planets_by_name.get(rule1_result.get("sixth_house_lord", ""), {}), gender, include_rashi=False)
    if source:
        sources.append(source)

    source = _build_source("P5", "Ascendant Lord", planets_by_name.get(rule1_result.get("ascendant_lord", ""), {}), gender, include_rashi=True)
    if source:
        sources.append(source)

    if include_moon:
        source = _build_source("PM", "Conditional Moon", moon_planet_data, gender, include_rashi=True)
        if source:
            source["connected_role"] = conditional_moon.get("connected_role", "")
            sources.append(source)

    top_disease_map: dict[str, dict] = {}

    for source in sources:
        source_weight = SOURCE_WEIGHTS.get(source["priority"], 1.0)
        layer_bundles = [
            ("planet", source.get("planet_organs", [])),
            ("nakshatra", source.get("nakshatra_organs", [])),
            ("source_rashi", source.get("rashi_organs", [])),
        ]

        for layer_name, organs in layer_bundles:
            if not organs:
                continue
            layer_weight = LAYER_WEIGHTS.get(layer_name, 1.0)
            if source["priority"] == "P2" and not has_occupant and layer_name == "source_rashi":
                layer_weight *= P2_ABSENT_OCCUPANT_RASHI_MULTIPLIER

            for organ in organs:
                allow_direct_p2_fallback = (
                    source["priority"] == "P2"
                    and not has_occupant
                    and layer_name == "source_rashi"
                )
                if organ not in anchor_organs and not allow_direct_p2_fallback:
                    continue
                total_weight = source_weight * layer_weight
                entry = organ_scores.setdefault(organ, _new_organ_entry(organ))
                entry["score"] += total_weight
                entry["exact_sources"].add(source["priority"])
                entry["trail"].append({
                    "source": source["priority"],
                    "label": source["label"],
                    "layer": layer_name,
                    "weight": round(total_weight, 3),
                })

        for disease in source.get("nakshatra_diseases", []):
            disease_organs = [o for o in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []) if _term_allowed(o, gender)]
            if not disease_organs:
                continue

            allow_direct_p2_disease_fallback = (
                source["priority"] == "P2"
                and not has_occupant
                and bool(source.get("rashi_organs"))
            )
            eligible_organs = set(anchor_organs)
            if allow_direct_p2_disease_fallback:
                eligible_organs.update(source.get("rashi_organs", []))

            hit_organs = [organ for organ in disease_organs if organ in eligible_organs]
            if not hit_organs:
                continue

            for organ in hit_organs:
                disease_weight = source_weight * LAYER_WEIGHTS["nakshatra"]

                entry = organ_scores.setdefault(organ, _new_organ_entry(organ))
                _add_disease_to_organ(entry, disease, disease_weight, source)
                disease_entry = entry["related_diseases"][disease]

                matched_planet_organs = [
                    item for item in source.get("planet_organs", [])
                    if item in disease_organs
                ]
                if matched_planet_organs and source.get("planet"):
                    support_entry = disease_entry["planet_support"].setdefault(source["planet"], {
                        "planet": source["planet"],
                        "priority": source["priority"],
                        "matched_organs": set(),
                    })
                    support_entry["matched_organs"].update(matched_planet_organs)

                disease_entry = top_disease_map.setdefault(disease, {
                    "disease": disease,
                    "score": 0.0,
                    "matched_organs": set(),
                    "source_priorities": set(),
                    "source_labels": set(),
                })
                disease_entry["score"] += disease_weight
                disease_entry["matched_organs"].add(organ)
                disease_entry["source_priorities"].add(source["priority"])
                disease_entry["source_labels"].add(source["label"])

    from eleventh_lord_confirmation import (
        confirmation_boost,
        find_disease_confirmation,
        get_eleventh_lord_confirmation_source,
    )
    confirmation_source = get_eleventh_lord_confirmation_source(
        chart_data, rule1_result, gender
    )
    confirmations = []
    for disease, disease_data in top_disease_map.items():
        match = find_disease_confirmation(disease, confirmation_source)
        if not match:
            continue
        boost = confirmation_boost(disease_data["score"])
        if boost <= 0:
            continue
        disease_data["score"] += boost
        disease_data["eleventh_lord_confirmation"] = {**match, "boost": round(boost, 2)}
        matched_organs = set(disease_data["matched_organs"])
        for organ in matched_organs:
            organ_entry = organ_scores.get(organ)
            if not organ_entry:
                continue
            organ_boost = confirmation_boost(organ_entry["score"])
            organ_entry["score"] += organ_boost
            related = organ_entry["related_diseases"].get(disease)
            if related:
                related["score"] += boost
                related["eleventh_lord_confirmation"] = {
                    **match,
                    "boost": round(boost, 2),
                }
            organ_entry["trail"].append({
                "source": "11L_NK_CONFIRM",
                "label": match["label"],
                "layer": "nakshatra_confirmation",
                "weight": round(organ_boost, 3),
            })
        confirmations.append({"disease": disease, **match, "boost": round(boost, 2)})

    ranked_organs = []
    for organ, data in organ_scores.items():
        related_diseases = []
        for disease_data in data["related_diseases"].values():
            planet_support = []
            for support_data in disease_data.get("planet_support", {}).values():
                planet_support.append({
                    "planet": support_data["planet"],
                    "priority": support_data["priority"],
                    "matched_organs": sorted(support_data["matched_organs"]),
                })
            planet_support.sort(key=lambda item: (item["planet"] != "Moon", item["planet"]))

            related_diseases.append({
                "disease": disease_data["disease"],
                "score": round(disease_data["score"], 2),
                "source_priorities": sorted(disease_data["source_priorities"]),
                "source_labels": sorted(disease_data["source_labels"]),
                "planet_support": planet_support,
                "eleventh_lord_confirmation": disease_data.get("eleventh_lord_confirmation"),
            })
        related_diseases.sort(key=lambda item: item["score"], reverse=True)

        ranked_organs.append({
            "organ": organ,
            "score": round(data["score"], 2),
            "anchor_hits": sorted(data["anchor_hits"]),
            "exact_sources": sorted(data["exact_sources"]),
            "disease_sources": sorted(data["disease_sources"]),
            "related_diseases": related_diseases,
            "trail": sorted(data["trail"], key=lambda item: item["weight"], reverse=True),
        })
    ranked_organs = [item for item in ranked_organs if _should_keep_ranked_organ(item)]
    ranked_organs.sort(key=lambda item: item["score"], reverse=True)

    ranked_diseases = []
    for disease, data in top_disease_map.items():
        ranked_diseases.append({
            "disease": disease,
            "score": round(data["score"], 2),
            "matched_organs": sorted(data["matched_organs"]),
            "source_priorities": sorted(data["source_priorities"]),
            "source_labels": sorted(data["source_labels"]),
            "eleventh_lord_confirmation": data.get("eleventh_lord_confirmation"),
        })
    ranked_diseases.sort(key=lambda item: item["score"], reverse=True)

    return {
        "anchors": anchors,
        "sources": sources,
        "top_organs": ranked_organs[:8],
        "top_diseases": ranked_diseases[:8],
        "all_organs": ranked_organs,
        "all_diseases": ranked_diseases,
        "eleventh_lord_nakshatra_confirmation": {
            "source": confirmation_source,
            "matches": confirmations,
        },
        "moon_included": include_moon,
        "moon_reason": conditional_moon.get("reason", ""),
        "fallback_note": "11th house occupant/rashi used as first priority because 6th house occupant is absent." if not has_occupant else "",
    }
