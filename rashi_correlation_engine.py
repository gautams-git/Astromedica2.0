from knowledge_base import (
    PLANET_DISEASES,
    NAKSHATRA_DISEASES,
    RASHI_ORGANS,
    is_term_allowed_for_gender,
    normalize_gender,
)
from term_to_systems import TERM_TO_SYSTEMS, MERGE_GROUPS


PRIORITY_WEIGHTS = {
    "P1": 6.0,
    "P2": 5.0,
    "P3": 3.0,
    "P4": 2.5,
    "P5": 2.0,
    "PM1": 6.0,
    "PM2": 4.0,
    "PM3": 3.0,
    "PM4": 2.5,
}

LAYER_WEIGHTS = {
    "planet": 0.85,
    "nakshatra": 1.25,
    "source_rashi": 0.8,
}

ANCHOR_WEIGHTS = {
    "P6": 3.0,
    "P7": 1.5,
}

SKIP_PREFIXES = [
    "Further everything",
    "Further all",
]

GENERIC_SHARED_SYSTEMS = {
    "dermatological",
    "sensory",
    "systemic",
}

SUPPORT_ONLY_SYSTEMS = {
    "blood",
    "lymphatic",
}


def _norm(value):
    return (value or "").strip().lower()


def _dedupe(items):
    seen = set()
    output = []
    for item in items:
        key = _norm(item)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _term_allowed(term, gender):
    return is_term_allowed_for_gender(term, gender)


def _planet_terms(planet_name):
    raw = PLANET_DISEASES.get(planet_name, "")
    terms = []
    for term in raw.split(","):
        clean = term.strip()
        if not clean:
            continue
        if any(clean.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        terms.append(clean)
    return terms


def _nakshatra_terms(nakshatra_name, pada):
    nk_data = NAKSHATRA_DISEASES.get(nakshatra_name, {})
    if not nk_data:
        return []
    pada_str = str(pada)
    for key, diseases in nk_data.items():
        padas_in_key = [p.strip() for p in key.split(",")]
        if pada_str in padas_in_key:
            return diseases
    return []


def _rashi_terms(rashi_name):
    raw = RASHI_ORGANS.get(rashi_name, "")
    return [term.strip() for term in raw.split(",") if term.strip()]


def _group_lookup():
    lookup = {}
    for group_id, group_data in MERGE_GROUPS.items():
        for system in group_data.get("systems", []):
            lookup[system] = {
                "id": group_id,
                "label": group_data.get("label", system),
            }
    return lookup


GROUP_LOOKUP = _group_lookup()


def _systems_for_terms(terms):
    systems = set()
    for term in terms:
        for system in TERM_TO_SYSTEMS.get(term, []):
            systems.add(system)
    return sorted(systems)


def _is_generic_only_match(shared_systems):
    if not shared_systems:
        return False
    return all(system in GENERIC_SHARED_SYSTEMS for system in shared_systems)


def _is_support_only_match(shared_systems):
    if not shared_systems:
        return False
    return all(system in SUPPORT_ONLY_SYSTEMS for system in shared_systems)


def _build_evidence_sources(chart_data, rule1_result, gender=None):
    planets_by_name = {planet.get("name"): planet for planet in chart_data.get("planets", [])}
    sources = []

    def add_source(priority, label, planet_name="", include_rashi=False):
        if not planet_name:
            return
        planet_data = planets_by_name.get(planet_name, {}) or {}
        nakshatra = (planet_data.get("nakshatra") or {}).get("name", "")
        pada = (planet_data.get("nakshatra") or {}).get("pada", 1)
        rashi = (planet_data.get("sign") or {}).get("name", "")

        planet_terms = [t for t in _planet_terms(planet_name) if _term_allowed(t, gender)]
        nak_terms = [t for t in _nakshatra_terms(nakshatra, pada) if _term_allowed(t, gender)]
        rashi_terms = [t for t in _rashi_terms(rashi) if _term_allowed(t, gender)] if include_rashi else []

        sources.append({
            "priority": priority,
            "label": label,
            "planet": planet_name,
            "nakshatra": nakshatra,
            "pada": pada,
            "rashi": rashi,
            "planet_terms": _dedupe(planet_terms),
            "nakshatra_terms": _dedupe(nak_terms),
            "rashi_terms": _dedupe(rashi_terms),
        })

    for rk in rule1_result.get("rogkaraka_1", []) or []:
        add_source("P1", "Occupant of 6th House", rk.get("name", ""), include_rashi=False)

    if not rule1_result.get("rogkaraka_1"):
        for rk in rule1_result.get("eleventh_house_occupants", []) or []:
            add_source("P1", "11th House Occupant as P1", rk.get("name", ""), include_rashi=False)

    rk3 = rule1_result.get("rogkaraka_3") or {}
    add_source("P2", "Dispositor of 6th Lord", rk3.get("name", ""), include_rashi=True)

    for drishti in rule1_result.get("drishti_on_6th", []) or []:
        add_source("P3", "Drishti on 6th House", drishti.get("planet", ""), include_rashi=True)

    add_source("P4", "6th House Lord", rule1_result.get("sixth_house_lord", ""), include_rashi=True)
    add_source("P5", "Ascendant Lord", rule1_result.get("ascendant_lord", ""), include_rashi=True)

    conditional_moon = rule1_result.get("conditional_moon") or {}
    if conditional_moon.get("include"):
        moon_role = conditional_moon.get("connected_role", "")
        moon_priority = {
            "occupant": "PM1",
            "dispositor": "PM2",
            "drishti": "PM3",
            "6th_house_lord": "PM4",
        }.get(moon_role, "PM4")
        add_source(moon_priority, f"Conditional Moon via {moon_role.replace('_', ' ')}", "Moon", include_rashi=True)

    return sources


def run_rashi_correlation_analysis(chart_data, rule1_result, gender=None):
    gender = normalize_gender(gender or chart_data.get("gender"))
    sixth_sign = rule1_result.get("sixth_house_sign", "")
    asc_sign = (chart_data.get("ascendant") or {}).get("name", "")

    anchors = {
        "P6": {
            "priority": "P6",
            "label": "6th House Rashi",
            "sign": sixth_sign,
            "terms": _dedupe([t for t in _rashi_terms(sixth_sign) if _term_allowed(t, gender)]),
        },
        "P7": {
            "priority": "P7",
            "label": "Ascendant Rashi",
            "sign": asc_sign,
            "terms": _dedupe([t for t in _rashi_terms(asc_sign) if _term_allowed(t, gender)]),
        }
    }
    for anchor in anchors.values():
        anchor["systems"] = _systems_for_terms(anchor["terms"])

    evidence_sources = _build_evidence_sources(chart_data, rule1_result, gender=gender)
    group_scores = {}

    for source in evidence_sources:
        source_priority = source.get("priority", "")
        priority_weight = PRIORITY_WEIGHTS.get(source_priority, 1.0)

        layer_items = [
            ("planet", source.get("planet_terms", [])),
            ("nakshatra", source.get("nakshatra_terms", [])),
            ("source_rashi", source.get("rashi_terms", [])),
        ]

        for anchor_key, anchor in anchors.items():
            anchor_weight = ANCHOR_WEIGHTS.get(anchor_key, 1.0)
            anchor_terms = anchor.get("terms", [])
            if not anchor_terms:
                continue

            for layer_name, terms in layer_items:
                if not terms:
                    continue
                layer_weight = LAYER_WEIGHTS.get(layer_name, 1.0)

                for evidence_term in terms:
                    evidence_systems = set(TERM_TO_SYSTEMS.get(evidence_term, []))
                    if not evidence_systems:
                        continue
                    best_match = None
                    for anchor_term in anchor_terms:
                        anchor_systems = set(TERM_TO_SYSTEMS.get(anchor_term, []))
                        if not anchor_systems:
                            continue

                        shared_systems = evidence_systems & anchor_systems
                        if not shared_systems and _norm(evidence_term) != _norm(anchor_term):
                            continue

                        exact_match = _norm(evidence_term) == _norm(anchor_term)
                        generic_only = _is_generic_only_match(shared_systems)
                        support_only = _is_support_only_match(shared_systems)

                        # Broad cosmetic/sensory/systemic overlaps were creating
                        # false positives. Unless the terms match exactly, do not
                        # let those overlaps create primary ranking.
                        if not exact_match and generic_only:
                            continue

                        # Blood/lymph-type overlaps are allowed only as support
                        # from disease-style layers, not from broad organ layers.
                        if support_only and layer_name != "nakshatra" and not exact_match:
                            continue

                        match_strength = 1.35 if exact_match else 1.0
                        if shared_systems:
                            match_strength += min(0.25, 0.08 * len(shared_systems))

                        candidate_systems = sorted(shared_systems or evidence_systems or anchor_systems)
                        for system in candidate_systems:
                            group_meta = GROUP_LOOKUP.get(system)
                            if not group_meta:
                                continue
                            if not exact_match and system in GENERIC_SHARED_SYSTEMS:
                                continue
                            if support_only and system in SUPPORT_ONLY_SYSTEMS and layer_name != "nakshatra" and not exact_match:
                                continue

                            specificity_bonus = 1.0 / max(len(shared_systems) if shared_systems else 1, 1)
                            score = priority_weight * layer_weight * anchor_weight * match_strength * (1.0 + specificity_bonus * 0.2)
                            candidate = {
                                "group_id": group_meta["id"],
                                "group_label": group_meta["label"],
                                "anchor_term": anchor_term,
                                "shared_system": system,
                                "match_type": "exact_term" if exact_match else "shared_system",
                                "weight": round(score, 3),
                                "priority_tuple": (
                                    1 if layer_name == "nakshatra" else 0,
                                    1 if anchor_key == "P6" else 0,
                                    1 if exact_match else 0,
                                    -len(shared_systems) if shared_systems else -99,
                                    round(score, 3),
                                ),
                            }
                            if not best_match or candidate["priority_tuple"] > best_match["priority_tuple"]:
                                best_match = candidate

                    if not best_match:
                        continue

                    group_id = best_match["group_id"]
                    group = group_scores.setdefault(group_id, {
                        "id": group_id,
                        "label": best_match["group_label"],
                        "score": 0.0,
                        "matched_anchors": set(),
                        "supporting_priorities": set(),
                        "matched_organs": set(),
                        "matched_diseases": set(),
                        "trail": [],
                    })

                    group["score"] += best_match["weight"]
                    group["matched_anchors"].add(anchor_key)
                    group["supporting_priorities"].add(source_priority)
                    group["matched_organs"].add(best_match["anchor_term"])
                    group["matched_diseases"].add(evidence_term)
                    group["trail"].append({
                        "priority": source_priority,
                        "source_label": source.get("label", ""),
                        "planet": source.get("planet", ""),
                        "layer": layer_name,
                        "evidence_term": evidence_term,
                        "anchor_priority": anchor_key,
                        "anchor_label": anchor.get("label", ""),
                        "anchor_sign": anchor.get("sign", ""),
                        "anchor_term": best_match["anchor_term"],
                        "shared_system": best_match["shared_system"],
                        "match_type": best_match["match_type"],
                        "weight": best_match["weight"],
                    })

    from eleventh_lord_confirmation import (
        confirmation_boost,
        find_disease_confirmation,
        get_eleventh_lord_confirmation_source,
    )
    confirmation_source = get_eleventh_lord_confirmation_source(
        chart_data, rule1_result, gender
    )
    confirmations = []
    for group in group_scores.values():
        best_match = None
        for disease in group["matched_diseases"]:
            match = find_disease_confirmation(disease, confirmation_source)
            if match:
                best_match = match
                break
        if not best_match:
            continue
        boost = confirmation_boost(group["score"])
        if boost <= 0:
            continue
        group["score"] += boost
        evidence = {**best_match, "boost": round(boost, 2)}
        group["trail"].append({
            "priority": "11L_NK_CONFIRM",
            "source_label": evidence["label"],
            "planet": confirmation_source.get("planet", ""),
            "layer": "nakshatra_confirmation",
            "evidence_term": evidence["confirmation_disease"],
            "anchor_priority": "",
            "anchor_label": "",
            "anchor_sign": "",
            "anchor_term": "",
            "shared_system": "",
            "match_type": evidence["match_type"],
            "weight": round(boost, 3),
        })
        group["eleventh_lord_confirmation"] = evidence
        confirmations.append({"group": group["label"], **evidence})

    all_groups = []
    for group in group_scores.values():
        matched_anchor_labels = []
        if "P6" in group["matched_anchors"]:
            matched_anchor_labels.append(f"P6 ({anchors['P6']['sign']})")
        if "P7" in group["matched_anchors"]:
            matched_anchor_labels.append(f"P7 ({anchors['P7']['sign']})")
        all_groups.append({
            "id": group["id"],
            "label": group["label"],
            "score": round(group["score"], 2),
            "matched_anchors": matched_anchor_labels,
            "supporting_priorities": sorted(group["supporting_priorities"]),
            "matched_organs": sorted(group["matched_organs"]),
            "matched_diseases": sorted(group["matched_diseases"]),
            "eleventh_lord_confirmation": group.get("eleventh_lord_confirmation"),
            "trail": sorted(group["trail"], key=lambda item: item["weight"], reverse=True),
            "reason": (
                f"Matched against {', '.join(matched_anchor_labels)} using "
                f"{', '.join(sorted(group['supporting_priorities']))} evidence."
            ),
        })

    all_groups.sort(key=lambda item: item["score"], reverse=True)

    return {
        "anchors": anchors,
        "evidence_sources": evidence_sources,
        "top3": all_groups[:3],
        "all_groups": all_groups,
        "eleventh_lord_nakshatra_confirmation": {
            "source": confirmation_source,
            "matches": confirmations,
        },
        "fallback_note": "11th house occupant/rashi used as first priority because 6th house occupant is absent." if not rule1_result.get("rogkaraka_1") else "",
    }
