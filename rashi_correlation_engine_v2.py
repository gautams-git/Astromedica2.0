from __future__ import annotations

"""
Rashi Correlation v2 -- new priority cascade, AND the structural fix: the
old engine hard-discarded any match that only shared a "generic" system
(dermatological/sensory/systemic) unless the term matched exactly -- since
almost every body-part term carries "dermatological" as a catch-all tag,
that rule made Skin (and anything else routed only through a generic
system) nearly invisible. v2 keeps exact matches at full weight but turns
generic-system-only matches into a *reduced-weight* signal instead of a
hard discard, so real evidence is never thrown away outright.
Fully isolated from rashi_correlation_engine.py.
"""

from knowledge_base import is_term_allowed_for_gender, normalize_gender
from term_to_systems import TERM_TO_SYSTEMS, MERGE_GROUPS
from priority_cascade import build_cascade_tiers, build_drishti_catalysts, catalyst_multiplier, rashi_terms

ANCHOR_WEIGHTS = {"P6": 3.0, "P7": 1.5}
GENERIC_SHARED_SYSTEMS = {"dermatological", "sensory", "systemic"}
GENERIC_MATCH_PENALTY = 0.5  # reduced weight instead of a hard discard


def _term_allowed(term: str, gender: str | None) -> bool:
    return is_term_allowed_for_gender(term, gender)


def _norm(value: str) -> str:
    return str(value or "").strip().lower()


def _group_lookup():
    lookup = {}
    for group_id, group_data in MERGE_GROUPS.items():
        for system in group_data.get("systems", []):
            lookup[system] = {"id": group_id, "label": group_data.get("label", system)}
    return lookup


GROUP_LOOKUP = _group_lookup()


def _anchors(chart_data: dict, rule1_result: dict, gender: str | None) -> dict:
    sixth_sign = rule1_result.get("sixth_house_sign", "")
    asc_sign = (chart_data.get("ascendant") or {}).get("name", "")
    return {
        "P6": {"label": "6th House Rashi", "sign": sixth_sign, "terms": [t for t in rashi_terms(sixth_sign) if _term_allowed(t, gender)]},
        "P7": {"label": "Ascendant Rashi", "sign": asc_sign, "terms": [t for t in rashi_terms(asc_sign) if _term_allowed(t, gender)]},
    }


def run_rashi_correlation_analysis_v2(chart_data: dict, rule1_result: dict, gender: str | None = None) -> dict:
    gender = normalize_gender(gender or chart_data.get("gender"))

    tiers = build_cascade_tiers(chart_data, rule1_result, gender=gender)
    catalysts = build_drishti_catalysts(chart_data, rule1_result, gender=gender)
    anchors = _anchors(chart_data, rule1_result, gender)

    group_scores: dict[str, dict] = {}

    for tier in tiers:
        multiplier, _matched = catalyst_multiplier(tier, catalysts)
        evidence_terms = set()
        for planet in tier["planets"]:
            evidence_terms.update(planet.get("planet_terms", []))
            evidence_terms.update(planet.get("nakshatra_diseases", []))
            evidence_terms.update(planet.get("rashi_organs", []))

        for anchor_key, anchor in anchors.items():
            anchor_weight = ANCHOR_WEIGHTS.get(anchor_key, 1.0)
            if not anchor["terms"]:
                continue

            for evidence_term in evidence_terms:
                evidence_systems = set(TERM_TO_SYSTEMS.get(evidence_term, []))
                if not evidence_systems:
                    continue

                for anchor_term in anchor["terms"]:
                    exact_match = _norm(evidence_term) == _norm(anchor_term)
                    anchor_systems = set(TERM_TO_SYSTEMS.get(anchor_term, []))
                    shared_systems = evidence_systems & anchor_systems
                    if not exact_match and not shared_systems:
                        continue

                    generic_only = bool(shared_systems) and all(s in GENERIC_SHARED_SYSTEMS for s in shared_systems)
                    match_penalty = 1.0 if (exact_match or not generic_only) else GENERIC_MATCH_PENALTY

                    candidate_systems = sorted(shared_systems) if shared_systems else sorted(evidence_systems & anchor_systems)
                    if not candidate_systems:
                        candidate_systems = sorted(evidence_systems) if exact_match else []

                    for system in candidate_systems:
                        group_meta = GROUP_LOOKUP.get(system)
                        if not group_meta:
                            continue
                        score = tier["weight"] * multiplier * anchor_weight * match_penalty * (1.3 if exact_match else 1.0)

                        group = group_scores.setdefault(group_meta["id"], {
                            "id": group_meta["id"], "label": group_meta["label"], "score": 0.0,
                            "tier_labels": set(), "matched_anchors": set(), "trail": [],
                        })
                        group["score"] += score
                        group["tier_labels"].add(tier["label"])
                        group["matched_anchors"].add(anchor_key)
                        group["trail"].append({
                            "tier": tier["label"],
                            "evidence_term": evidence_term,
                            "anchor": anchor_key,
                            "anchor_term": anchor_term,
                            "match_type": "exact_term" if exact_match else ("generic_system" if generic_only else "shared_system"),
                            "weight": round(score, 3),
                        })

    all_groups = []
    for group in group_scores.values():
        all_groups.append({
            "id": group["id"],
            "label": group["label"],
            "score": round(group["score"], 2),
            "tier_labels": sorted(group["tier_labels"]),
            "matched_anchors": sorted(group["matched_anchors"]),
            "trail": sorted(group["trail"], key=lambda item: item["weight"], reverse=True),
        })
    all_groups.sort(key=lambda item: item["score"], reverse=True)

    return {
        "title": "Rashi Correlation v2",
        "subtitle": "Generic-system-only matches (e.g. dermatological) are down-weighted, not discarded -- so organs like Skin can still surface on real evidence.",
        "anchors": anchors,
        "top3": all_groups[:3],
        "all_groups": all_groups,
    }
