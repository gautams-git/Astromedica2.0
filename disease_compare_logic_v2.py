from __future__ import annotations

"""
Disease Compare Logic v2 -- same priority cascade as disease_first_logic_v2,
but a candidate disease only counts if it is a very-close wording match to a
disease actually produced by a cascade tier AND shares an organ with it.
Fully isolated from disease_compare_logic.py.
"""

from difflib import SequenceMatcher

from knowledge_base import is_term_allowed_for_gender, normalize_gender
from rashi_truth_review import NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS
from priority_cascade import build_cascade_tiers, build_drishti_catalysts, catalyst_multiplier

GENERIC_TOKENS = {"disease", "diseases", "disorder", "disorders", "problem", "problems", "trouble", "troubles"}


def _term_allowed(term: str, gender: str | None) -> bool:
    return is_term_allowed_for_gender(term, gender)


def _norm_text(value: str) -> str:
    text = str(value or "").strip().lower()
    for old, new in {"&": " and ", "/": " ", "-": " ", "(": " ", ")": " ", ",": " ", ".": " ", "'": ""}.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def _meaningful_tokens(value: str) -> set[str]:
    return {t for t in _norm_text(value).split() if t and t not in GENERIC_TOKENS}


def _is_very_close_disease(source_disease: str, candidate_disease: str) -> tuple[bool, float]:
    left, right = _norm_text(source_disease), _norm_text(candidate_disease)
    if not left or not right:
        return False, 0.0
    if left == right:
        return True, 1.0
    if left in right or right in left:
        return True, round(SequenceMatcher(None, left, right).ratio(), 4)

    ratio = SequenceMatcher(None, left, right).ratio()
    shared = _meaningful_tokens(source_disease) & _meaningful_tokens(candidate_disease)
    if ratio >= 0.9:
        return True, round(ratio, 4)
    if shared and ratio >= 0.82:
        return True, round(ratio, 4)
    left_tokens, right_tokens = _meaningful_tokens(source_disease), _meaningful_tokens(candidate_disease)
    if left_tokens and right_tokens:
        coverage = len(shared) / max(1, min(len(left_tokens), len(right_tokens)))
        if coverage >= 0.75 and ratio >= 0.72:
            return True, round(ratio, 4)
    return False, round(ratio, 4)


def _disease_organs(disease: str, gender: str | None) -> list[str]:
    return [o for o in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []) if _term_allowed(o, gender)]


def run_disease_compare_logic_v2(chart_data: dict, rule1_result: dict, gender: str | None = None) -> dict:
    gender = normalize_gender(gender or chart_data.get("gender"))

    tiers = build_cascade_tiers(chart_data, rule1_result, gender=gender)
    catalysts = build_drishti_catalysts(chart_data, rule1_result, gender=gender)
    all_candidate_diseases = list(NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.keys())

    disease_scores: dict[str, dict] = {}

    for tier in tiers:
        multiplier, matched_catalysts = catalyst_multiplier(tier, catalysts)

        for planet in tier["planets"]:
            for source_disease in planet.get("nakshatra_diseases", []):
                source_organs = _disease_organs(source_disease, gender)
                if not source_organs:
                    continue

                for candidate_disease in all_candidate_diseases:
                    is_close, similarity = _is_very_close_disease(source_disease, candidate_disease)
                    if not is_close:
                        continue
                    candidate_organs = _disease_organs(candidate_disease, gender)
                    shared_organs = sorted(set(source_organs) & set(candidate_organs))
                    if not shared_organs:
                        continue

                    entry = disease_scores.setdefault(candidate_disease, {
                        "disease": candidate_disease,
                        "score": 0.0,
                        "tier_labels": set(),
                        "matches": [],
                        "catalyst_matches": [],
                    })
                    entry["score"] += tier["weight"] * multiplier
                    entry["tier_labels"].add(tier["label"])
                    entry["matches"].append({
                        "source_disease": source_disease,
                        "matched_disease": candidate_disease,
                        "similarity": round(float(similarity), 3),
                        "shared_organs": shared_organs,
                        "tier": tier["label"],
                    })
                    if matched_catalysts:
                        entry["catalyst_matches"].append({"tier": tier["label"], "matches": matched_catalysts})

    ranked = []
    for disease, entry in disease_scores.items():
        ranked.append({
            "disease": disease,
            "score": round(entry["score"], 2),
            "tier_labels": sorted(entry["tier_labels"]),
            "source_disease_matches": entry["matches"],
            "catalyst_matches": entry["catalyst_matches"],
        })
    ranked.sort(key=lambda item: item["score"], reverse=True)

    return {
        "title": "Disease Compare Logic v2",
        "subtitle": "Same cascade as Disease First v2, but only very-close disease matches with shared organs are counted.",
        "top_diseases": ranked[:8],
        "all_diseases": ranked,
    }
