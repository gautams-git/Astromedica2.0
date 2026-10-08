from __future__ import annotations

"""
Disease-First Logic v2 -- uses the new priority cascade (priority_cascade.py)
instead of the old fixed P1-P7 ladder. Drishti never contributes its own
diseases; it only multiplies whichever cascade tier it shares evidence with.
Fully isolated from disease_first_logic.py (no shared imports of that file).
"""

from knowledge_base import is_term_allowed_for_gender, normalize_gender
from rashi_truth_review import ALL_ORGANS, NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS
from priority_cascade import build_cascade_tiers, build_drishti_catalysts, catalyst_multiplier

ALL_ORGANS_LOOKUP = {str(item).strip().lower(): item for item in ALL_ORGANS}


def _term_allowed(term: str, gender: str | None) -> bool:
    return is_term_allowed_for_gender(term, gender)


def _canonical_organ(term: str):
    return ALL_ORGANS_LOOKUP.get(str(term or "").strip().lower())


def _new_disease_entry(disease: str) -> dict:
    return {
        "disease": disease,
        "score": 0.0,
        "tier_labels": set(),
        "supporting_organs": {},
        "catalyst_matches": [],
    }


def _add_organ_support(entry: dict, organ: str, score: float, tier_label: str) -> None:
    organ_entry = entry["supporting_organs"].setdefault(organ, {"organ": organ, "score": 0.0, "supports": []})
    organ_entry["score"] += score
    organ_entry["supports"].append({"source": tier_label, "score": round(score, 3)})


def run_disease_first_logic_v2(chart_data: dict, rule1_result: dict, gender: str | None = None) -> dict:
    gender = normalize_gender(gender or chart_data.get("gender"))

    tiers = build_cascade_tiers(chart_data, rule1_result, gender=gender)
    catalysts = build_drishti_catalysts(chart_data, rule1_result, gender=gender)

    disease_scores: dict[str, dict] = {}

    for tier in tiers:
        multiplier, matched_catalysts = catalyst_multiplier(tier, catalysts)

        for planet in tier["planets"]:
            for disease in planet.get("nakshatra_diseases", []):
                disease_organs = [o for o in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []) if _term_allowed(o, gender)]
                if not disease_organs:
                    continue

                entry = disease_scores.setdefault(disease, _new_disease_entry(disease))
                entry["score"] += tier["weight"] * multiplier
                entry["tier_labels"].add(tier["label"])
                if matched_catalysts:
                    entry["catalyst_matches"].append({"tier": tier["label"], "matches": matched_catalysts})

                matched_planet_organs = [
                    _canonical_organ(t) for t in planet.get("planet_terms", [])
                    if _canonical_organ(t) in disease_organs
                ]
                for organ in matched_planet_organs:
                    _add_organ_support(entry, organ, tier["planet_layer_weight"] * multiplier, tier["label"])
                    entry["score"] += tier["planet_layer_weight"] * multiplier

                matched_rashi_organs = [o for o in planet.get("rashi_organs", []) if o in disease_organs]
                for organ in matched_rashi_organs:
                    _add_organ_support(entry, organ, tier["rashi_layer_weight"] * multiplier, f"{tier['label']} Rashi")
                    entry["score"] += tier["rashi_layer_weight"] * multiplier

    ranked = []
    for disease, entry in disease_scores.items():
        supporting_organs = sorted(
            (
                {"organ": data["organ"], "score": round(data["score"], 2), "supports": data["supports"]}
                for data in entry["supporting_organs"].values()
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        ranked.append({
            "disease": disease,
            "score": round(entry["score"], 2),
            "tier_labels": sorted(entry["tier_labels"]),
            "supporting_organs": supporting_organs,
            "catalyst_matches": entry["catalyst_matches"],
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)

    return {
        "title": "Disease First Logic v2",
        "subtitle": "Priority cascade (6th Occ -> 11th Occ -> 6th Lord -> Dispositor -> Asc/Moon); drishti acts only as a catalyst.",
        "cascade_tiers": [
            {"rank": t["rank"], "tier_id": t["tier_id"], "label": t["label"], "weight": t["weight"], "planets": [p["planet"] for p in t["planets"]]}
            for t in tiers
        ],
        "top_diseases": ranked[:8],
        "all_diseases": ranked,
    }
