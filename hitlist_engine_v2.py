from __future__ import annotations

"""
HitList Engine v2 -- same system/group merge approach (TERM_TO_SYSTEMS /
MERGE_GROUPS) as the original, but driven by the new priority cascade
instead of the fixed P1-P7 ladder. Drishti is a catalyst multiplier only,
never its own HitList layer. Fully isolated from hitlist_engine.py.
"""

from knowledge_base import is_term_allowed_for_gender, normalize_gender
from term_to_systems import TERM_TO_SYSTEMS, MERGE_GROUPS
from priority_cascade import build_cascade_tiers, build_drishti_catalysts, catalyst_multiplier

HL_MULTIPLIERS = {"HL1": 1.5, "HL2": 6.0, "HL3": 7.0}
CONVERGENCE = {3: 2.0, 2: 1.4, 1: 1.0}


def _term_allowed(term: str, gender: str | None) -> bool:
    return is_term_allowed_for_gender(term, gender)


def _score_layer(terms, weight, layer_id, system_scores, hitlist_tracker, term_trail, tier_label, gender):
    for term in terms:
        if not _term_allowed(term, gender):
            continue
        systems = TERM_TO_SYSTEMS.get(term, [])
        if not systems:
            continue
        specificity = 1.0 / len(systems)
        term_weight = weight * HL_MULTIPLIERS[layer_id] * specificity
        for system in systems:
            system_scores[system] = system_scores.get(system, 0.0) + term_weight
            hitlist_tracker.setdefault(system, set()).add(layer_id)
            term_trail.setdefault(system, []).append({
                "term": term,
                "source": f"{layer_id} {tier_label}",
                "weight": round(term_weight, 3),
            })


def run_hitlist_analysis_v2(chart_data: dict, rule1_result: dict, gender: str | None = None) -> dict:
    gender = normalize_gender(gender or chart_data.get("gender"))

    tiers = build_cascade_tiers(chart_data, rule1_result, gender=gender)
    catalysts = build_drishti_catalysts(chart_data, rule1_result, gender=gender)

    system_scores: dict[str, float] = {}
    hitlist_tracker: dict[str, set] = {}
    term_trail: dict[str, list] = {}

    for tier in tiers:
        multiplier, _matched = catalyst_multiplier(tier, catalysts)
        effective_weight = tier["weight"] * multiplier
        for planet in tier["planets"]:
            _score_layer(planet.get("planet_terms", []), effective_weight, "HL1", system_scores, hitlist_tracker, term_trail, tier["label"], gender)
            _score_layer(planet.get("nakshatra_diseases", []), effective_weight, "HL2", system_scores, hitlist_tracker, term_trail, tier["label"], gender)
            _score_layer(planet.get("rashi_organs", []), effective_weight, "HL3", system_scores, hitlist_tracker, term_trail, tier["label"], gender)

    final_scores = {}
    for system, raw_score in system_scores.items():
        hitlists_count = len(hitlist_tracker.get(system, set()))
        multiplier = CONVERGENCE.get(hitlists_count, 1.0)
        final_scores[system] = round(raw_score * multiplier, 2)

    group_scores = []
    for group_id, group_data in MERGE_GROUPS.items():
        systems_in_group = group_data["systems"]
        label = group_data["label"]
        total = sum(final_scores.get(s, 0.0) for s in systems_in_group)
        if total == 0:
            continue
        combined_hitlists = set()
        combined_trail = []
        for s in systems_in_group:
            combined_hitlists.update(hitlist_tracker.get(s, set()))
            combined_trail.extend(term_trail.get(s, []))
        group_scores.append({
            "label": label,
            "score": round(total, 2),
            "hitlists": sorted(combined_hitlists),
            "convergence": len(combined_hitlists),
            "trail": combined_trail,
        })

    group_scores.sort(key=lambda x: x["score"], reverse=True)

    return {
        "title": "HitList Engine v2",
        "subtitle": "Priority cascade scoring; drishti acts only as a catalyst multiplier.",
        "cascade_tiers": [
            {"rank": t["rank"], "tier_id": t["tier_id"], "label": t["label"], "weight": t["weight"], "planets": [p["planet"] for p in t["planets"]]}
            for t in tiers
        ],
        "top4": group_scores[:4],
        "all_groups": group_scores,
    }
