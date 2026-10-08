from __future__ import annotations

from difflib import SequenceMatcher

from disease_first_logic import (
    PLANET_LAYER_WEIGHTS,
    RASHI_LAYER_WEIGHTS,
    SOURCE_WEIGHTS,
    _add_organ_support,
    _build_source,
    _dedupe,
    _new_disease_entry,
    _planet_data,
    _rashi_terms,
    _term_allowed,
)
from knowledge_base import normalize_gender
from rashi_truth_review import NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS


PRIORITY_SORT_ORDER = ["P1", "P2", "P3", "P4", "PM", "P5"]
RASHI_PRIORITY_SORT_ORDER = ["P1_RASHI", "P2_RASHI", "P6", "PM_RASHI", "P7"]
GENERIC_TOKENS = {
    "disease",
    "diseases",
    "disorder",
    "disorders",
    "problem",
    "problems",
    "trouble",
    "troubles",
}


def _norm_text(value: str) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "&": " and ",
        "/": " ",
        "-": " ",
        "(": " ",
        ")": " ",
        ",": " ",
        ".": " ",
        "'": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token
        for token in _norm_text(value).split()
        if token and token not in GENERIC_TOKENS
    }


def _is_very_close_disease(source_disease: str, candidate_disease: str) -> tuple[bool, str, float]:
    left = _norm_text(source_disease)
    right = _norm_text(candidate_disease)
    if not left or not right:
        return False, "", 0.0

    if left == right:
        return True, "Exact normalized disease match", 1.0

    if left in right or right in left:
        ratio = SequenceMatcher(None, left, right).ratio()
        return True, "Very close disease wording variant", round(ratio, 4)

    ratio = SequenceMatcher(None, left, right).ratio()
    left_tokens = _meaningful_tokens(source_disease)
    right_tokens = _meaningful_tokens(candidate_disease)
    shared = left_tokens & right_tokens

    if ratio >= 0.9:
        return True, "Very close disease wording variant", round(ratio, 4)

    if shared and ratio >= 0.82:
        return True, "Very close disease wording with shared medical tokens", round(ratio, 4)

    if left_tokens and right_tokens:
        coverage = len(shared) / max(1, min(len(left_tokens), len(right_tokens)))
        if coverage >= 0.75 and ratio >= 0.72:
            return True, "Very close disease wording with strong token overlap", round(ratio, 4)

    return False, "", round(ratio, 4)


def _disease_organs(disease: str, gender: str | None) -> list[str]:
    return [
        organ
        for organ in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, [])
        if _term_allowed(organ, gender)
    ]


def _candidate_match(source_disease: str, candidate_disease: str, gender: str | None) -> dict | None:
    is_close, reason, similarity = _is_very_close_disease(source_disease, candidate_disease)
    if not is_close:
        return None

    source_organs = _disease_organs(source_disease, gender)
    candidate_organs = _disease_organs(candidate_disease, gender)
    shared_organs = sorted({organ for organ in source_organs if organ in candidate_organs})
    if not shared_organs:
        return None

    return {
        "source_disease": source_disease,
        "matched_disease": candidate_disease,
        "source_organs": source_organs,
        "matched_organs": candidate_organs,
        "shared_organs": shared_organs,
        "reason": reason,
        "similarity": similarity,
    }


def run_disease_compare_logic(chart_data: dict, rule1_result: dict, gender: str | None = None) -> dict:
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
    all_candidate_diseases = list(NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.keys())

    for source in sources:
        for source_disease in source.get("diseases", []):
            for candidate_disease in all_candidate_diseases:
                match = _candidate_match(source_disease, candidate_disease, gender)
                if not match:
                    continue

                entry = disease_scores.setdefault(candidate_disease, _new_disease_entry(candidate_disease))
                entry.setdefault("source_disease_matches", [])
                entry.setdefault("matched_disease_reasons", [])

                base_weight = SOURCE_WEIGHTS.get(source["priority"], 1.0)
                entry["score"] += base_weight
                entry["source_priorities"].add(source["priority"])
                entry["source_labels"].add(source["label"])

                entry["source_disease_matches"].append({
                    "source_disease": source_disease,
                    "matched_disease": candidate_disease,
                    "priority": source["priority"],
                    "label": source["label"],
                    "reason": match["reason"],
                    "similarity": match["similarity"],
                    "shared_organs": match["shared_organs"],
                })

                reason_key = (
                    source_disease,
                    candidate_disease,
                    source["priority"],
                    ",".join(match["shared_organs"]),
                )
                if reason_key not in entry["matched_disease_reasons"]:
                    entry["matched_disease_reasons"].append(reason_key)

                matched_planet_organs = [o for o in source.get("planet_organs", []) if o in match["matched_organs"]]
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

                for organ in [o for o in source.get("rashi_organs", []) if o in match["matched_organs"]]:
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
            disease_organs = _disease_organs(disease, gender)
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
        planet_support.sort(
            key=lambda item: (
                PRIORITY_SORT_ORDER.index(item["priority"]) if item["priority"] in PRIORITY_SORT_ORDER else 99,
                item["planet"],
            )
        )

        rashi_support = []
        for support in entry["rashi_support"].values():
            rashi_support.append({
                "priority": support["priority"],
                "label": support["label"],
                "matched_organs": sorted(support["matched_organs"]),
            })
        rashi_support.sort(
            key=lambda item: (
                RASHI_PRIORITY_SORT_ORDER.index(item["priority"]) if item["priority"] in RASHI_PRIORITY_SORT_ORDER else 99,
                item["label"],
            )
        )

        source_matches = []
        seen_matches: set[tuple[str, str, str, str]] = set()
        for match in sorted(
            entry.get("source_disease_matches", []),
            key=lambda item: (
                PRIORITY_SORT_ORDER.index(item["priority"]) if item["priority"] in PRIORITY_SORT_ORDER else 99,
                -float(item["similarity"] or 0),
                item["source_disease"],
            ),
        ):
            key = (
                match["source_disease"],
                match["matched_disease"],
                match["priority"],
                ",".join(match["shared_organs"]),
            )
            if key in seen_matches:
                continue
            seen_matches.add(key)
            source_matches.append({
                "source_disease": match["source_disease"],
                "matched_disease": match["matched_disease"],
                "priority": match["priority"],
                "label": match["label"],
                "reason": match["reason"],
                "similarity": round(float(match["similarity"] or 0), 2),
                "shared_organs": match["shared_organs"],
            })

        ranked.append({
            "disease": disease,
            "score": round(entry["score"], 2),
            "source_priorities": sorted(entry["source_priorities"]),
            "source_labels": sorted(entry["source_labels"]),
            "supporting_organs": supporting_organs,
            "planet_support": planet_support,
            "rashi_support": rashi_support,
            "source_disease_matches": source_matches,
            "eleventh_lord_confirmation": entry.get("eleventh_lord_confirmation"),
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)

    return {
        "title": "Disease Compare Logic",
        "subtitle": "Same priority structure as disease-first logic, but disease-to-disease comparison happens first and only very-close matches with organ overlap continue.",
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
