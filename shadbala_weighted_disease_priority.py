from __future__ import annotations

"""
Shadbala-Weighted Disease Priority.

A separate, ADDITIONAL ranking -- independent of the anchored organ-overlap
cascade in disease_priority_cascade_logic_v2.py, which this file never
touches or reads from. That algorithm deliberately avoids weights/scores by
design (real organ evidence should never be out-ranked by an unrelated
signal); this one is the opposite on purpose: a disease's score is its
cascade tier's positional weight PLUS marks earned by how strong -- by
Shadbala, in THIS specific chart -- the planet that produced it is.

Client spec (confirmed 2026-07-08):
  - Marks are assigned by SHADBALA RANK POSITION within this chart (1st
    strongest planet through 7th weakest), not raw Rupas/Virupas numbers --
    reusing the same decay curve already used for cascade-tier weights
    elsewhere in the app: 6.0, 4.5, 3.5, 2.8, 2.0, then a 1.5 floor for
    any rank beyond that.
  - A disease's score = the positional weight of the cascade tier that
    produced it (6th Occupant / 11th Occupant / ... same decay curve) +
    the Shadbala marks of the SPECIFIC planet that produced it -- not the
    tier as a whole, since a tier can hold more than one occupant with
    different Shadbala strengths.
  - If the same disease is produced by more than one planet/tier, its
    score is the BEST (highest) of those paths; every contributing
    source is still kept for the backend-trail view.
  - Shown as its own section alongside (never replacing) the existing
    anchored Disease-Driven Priority Logic -- two different lenses on the
    same chart, not one overriding the other.
"""

from priority_cascade import build_cascade_tiers
from knowledge_base import is_term_allowed_for_gender, normalize_gender
from rashi_truth_review import NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS

# Marks by Shadbala rank position within this chart (client spec,
# 2026-07-08): Rank 1 = 12, Rank 2 = 8, then the same shrinking-drop
# pattern continued down to a floor of 1 for any rank beyond the schedule.
SHADBALA_MARKS_SCHEDULE = [12.0, 8.0, 5.0, 3.0, 2.0, 1.0]
SHADBALA_MARKS_FLOOR = 1.0


def _shadbala_marks_for_rank(rank: int) -> float:
    if rank <= len(SHADBALA_MARKS_SCHEDULE):
        return SHADBALA_MARKS_SCHEDULE[rank - 1]
    return SHADBALA_MARKS_FLOOR


def _disease_organs(disease: str, gender: str | None) -> list[str]:
    return [o for o in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []) if is_term_allowed_for_gender(o, gender)]


def compute_shadbala_marks(shadbala: dict) -> dict:
    """planet -> {"rank": int, "marks": float, "rupas": float|None, "virupas": float|None}.

    Ranked by Rupas (falling back to Virupas if Rupas is missing for a
    planet) descending -- rank 1 is the strongest planet in this chart.
    """
    shadbala = shadbala or {}
    rupas = shadbala.get("rupas") or {}
    virupas = shadbala.get("virupas") or {}
    planets = set(rupas) | set(virupas)
    if not planets:
        return {}

    def strength(planet: str) -> float:
        value = rupas.get(planet)
        if value is None:
            value = virupas.get(planet)
        return value if value is not None else float("-inf")

    ordered = sorted(planets, key=strength, reverse=True)
    result = {}
    for idx, planet in enumerate(ordered, start=1):
        result[planet] = {
            "rank": idx,
            "marks": _shadbala_marks_for_rank(idx),
            "rupas": rupas.get(planet),
            "virupas": virupas.get(planet),
        }
    return result


def run_shadbala_weighted_disease_priority(chart_data: dict, rule1_result: dict, shadbala: dict, gender: str | None = None) -> dict:
    gender = normalize_gender(gender or chart_data.get("gender"))
    tiers = build_cascade_tiers(chart_data, rule1_result, gender=gender)
    planet_marks = compute_shadbala_marks(shadbala)

    if not tiers:
        return {
            "title": "Shadbala-Weighted Disease Priority",
            "subtitle": "No cascade tiers resolved for this chart.",
            "planet_marks": planet_marks,
            "ranked_diseases": [],
        }

    disease_index: dict[str, dict] = {}
    for tier in tiers:
        tier_weight = tier["weight"]
        for planet in tier.get("planets", []):
            planet_name = planet.get("planet", "")
            marks_info = planet_marks.get(planet_name, {"rank": None, "marks": 0.0})
            score = round(tier_weight + marks_info["marks"], 2)
            for disease in planet.get("nakshatra_diseases", []):
                entry = disease_index.setdefault(disease, {
                    "disease": disease,
                    "disease_organs": _disease_organs(disease, gender),
                    "best_score": None,
                    "sources": [],
                })
                entry["sources"].append({
                    "tier_rank": tier["rank"],
                    "tier_label": tier["label"],
                    "tier_weight": tier_weight,
                    "planet": planet_name,
                    "nakshatra": planet.get("nakshatra", ""),
                    "pada": planet.get("pada"),
                    "nakshatra_diseases": planet.get("nakshatra_diseases", []),
                    "shadbala_rank": marks_info["rank"],
                    "shadbala_marks": marks_info["marks"],
                    "score": score,
                })
                if entry["best_score"] is None or score > entry["best_score"]:
                    entry["best_score"] = score

    ordered_diseases = sorted(disease_index.values(), key=lambda e: (-e["best_score"], e["disease"]))

    results = []
    for rank, entry in enumerate(ordered_diseases, start=1):
        sources = sorted(entry["sources"], key=lambda s: -s["score"])
        results.append({
            "rank": rank,
            "disease": entry["disease"],
            "disease_organs": entry["disease_organs"],
            "score": entry["best_score"],
            "best_source": sources[0],
            "sources": sources,
        })

    return {
        "title": "Shadbala-Weighted Disease Priority",
        "subtitle": (
            "Independent ranking: cascade tier position + the producing planet's "
            "Shadbala strength in this chart. Does not affect the anchored "
            "Disease-Driven Priority Logic above."
        ),
        "planet_marks": planet_marks,
        "ranked_diseases": results,
    }
