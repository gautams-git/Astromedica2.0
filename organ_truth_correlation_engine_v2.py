from __future__ import annotations

"""
Organ Truth Correlation v2 -- new priority cascade, AND the structural fix:
the old engine only scored an organ if it was literally present in the
6th-house/Ascendant rashi's organ list (anchor_organs), which made organs
like Skin mathematically impossible to surface (no rashi sign lists "Skin").
v2 scores every organ a tier's planet/nakshatra/rashi evidence actually
produces -- anchors (6th house rashi, ascendant rashi) now act as a
*confidence bonus* on top of that evidence, not a gate that excludes it.
Fully isolated from organ_truth_correlation_engine.py.
"""

from knowledge_base import is_term_allowed_for_gender, normalize_gender
from rashi_truth_review import ALL_ORGANS, NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS
from priority_cascade import (
    build_cascade_tiers,
    build_drishti_catalysts,
    catalyst_multiplier,
    rashi_terms,
)

ALL_ORGANS_LOOKUP = {str(item).strip().lower(): item for item in ALL_ORGANS}

ANCHOR_BONUS_RATIO = 0.5  # +50% bonus when an organ also matches a 6th/Asc rashi anchor


def _term_allowed(term: str, gender: str | None) -> bool:
    return is_term_allowed_for_gender(term, gender)


def _canonical_organ(term: str):
    return ALL_ORGANS_LOOKUP.get(str(term or "").strip().lower())


def _anchor_organs(chart_data: dict, rule1_result: dict, gender: str | None) -> set[str]:
    sixth_sign = rule1_result.get("sixth_house_sign", "")
    asc_sign = (chart_data.get("ascendant") or {}).get("name", "")
    organs: set[str] = set()
    for sign in (sixth_sign, asc_sign):
        for term in rashi_terms(sign):
            canonical = _canonical_organ(term)
            if canonical and _term_allowed(canonical, gender):
                organs.add(canonical)
    return organs


def _new_organ_entry(organ: str) -> dict:
    return {"organ": organ, "score": 0.0, "tier_labels": set(), "anchor_hit": False, "related_diseases": {}}


def run_organ_truth_correlation_analysis_v2(chart_data: dict, rule1_result: dict, gender: str | None = None) -> dict:
    gender = normalize_gender(gender or chart_data.get("gender"))

    tiers = build_cascade_tiers(chart_data, rule1_result, gender=gender)
    catalysts = build_drishti_catalysts(chart_data, rule1_result, gender=gender)
    anchor_organs = _anchor_organs(chart_data, rule1_result, gender)

    organ_scores: dict[str, dict] = {}

    def credit(organ: str, weight: float, tier_label: str):
        entry = organ_scores.setdefault(organ, _new_organ_entry(organ))
        bonus = (1.0 + ANCHOR_BONUS_RATIO) if organ in anchor_organs else 1.0
        entry["score"] += weight * bonus
        entry["tier_labels"].add(tier_label)
        if organ in anchor_organs:
            entry["anchor_hit"] = True

    for tier in tiers:
        multiplier, _matched = catalyst_multiplier(tier, catalysts)

        for planet in tier["planets"]:
            for term in planet.get("planet_terms", []):
                canonical = _canonical_organ(term)
                if canonical:
                    credit(canonical, tier["planet_layer_weight"] * multiplier, tier["label"])

            for organ in planet.get("rashi_organs", []):
                credit(organ, tier["rashi_layer_weight"] * multiplier, tier["label"])

            for disease in planet.get("nakshatra_diseases", []):
                disease_organs = [o for o in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []) if _term_allowed(o, gender)]
                if not disease_organs:
                    continue
                for organ in disease_organs:
                    entry = organ_scores.setdefault(organ, _new_organ_entry(organ))
                    bonus = (1.0 + ANCHOR_BONUS_RATIO) if organ in anchor_organs else 1.0
                    weight = tier["weight"] * multiplier * bonus
                    entry["score"] += weight
                    entry["tier_labels"].add(tier["label"])
                    if organ in anchor_organs:
                        entry["anchor_hit"] = True
                    disease_entry = entry["related_diseases"].setdefault(disease, {
                        "disease": disease, "score": 0.0, "tier_labels": set(),
                    })
                    disease_entry["score"] += weight
                    disease_entry["tier_labels"].add(tier["label"])

    ranked = []
    for organ, data in organ_scores.items():
        related = sorted(
            (
                {"disease": d["disease"], "score": round(d["score"], 2), "tier_labels": sorted(d["tier_labels"])}
                for d in data["related_diseases"].values()
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        ranked.append({
            "organ": organ,
            "score": round(data["score"], 2),
            "tier_labels": sorted(data["tier_labels"]),
            "anchor_hit": data["anchor_hit"],
            "related_diseases": related,
        })
    ranked.sort(key=lambda item: item["score"], reverse=True)

    return {
        "title": "Organ Truth Correlation v2",
        "subtitle": "No anchor-gating: every organ a cascade tier actually produces is scored. 6th-house/Ascendant rashi organs add a confidence bonus, not a filter.",
        "anchor_organs": sorted(anchor_organs),
        "top_organs": ranked[:8],
        "all_organs": ranked,
        "top_diseases": [],
    }
