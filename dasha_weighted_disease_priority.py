from __future__ import annotations

"""
Dasha-Weighted Disease Priority.

A separate, ADDITIONAL ranking -- independent of the anchored organ-overlap
cascade (disease_priority_cascade_logic_v2.py) and independent of the
Shadbala-weighted ranking (shadbala_weighted_disease_priority.py). Neither
of those is touched or read here. This one boosts diseases produced by
whichever planet is CURRENTLY running as Mahadasha or Antardasha for this
chart, on top of the usual cascade-tier positional rank.

Client spec (confirmed 2026-07-13):
  - A disease's score = the positional weight of the cascade tier that
    produced it (6th Occupant / 11th Occupant / 6th Lord / Dispositor /
    Asc-Moon -- same decay curve used everywhere else: 6.0, 4.5, 3.5, 2.8,
    2.0, floor 1.5) PLUS dasha marks for the SPECIFIC planet that produced
    it:
      - current Mahadasha planet  -> +12
      - current Antardasha planet -> +7
      - neither                   -> +0 (score is just the tier weight,
                                          i.e. the "usual rank")
    If a planet happens to be both Mahadasha and Antardasha at once (not
    possible under standard Vimshottari nesting, but guarded anyway), both
    add.
  - A planet already occupying one of the 5 hierarchy tiers gets its dasha
    marks added ON TOP of that tier's weight -- e.g. an 11th House
    Occupant (tier weight 4.5) that is also the current Mahadasha planet
    scores 4.5 + 12 = 16.5, which can out-rank a 6th House Occupant
    (weight 6.0, tier weight only) that has no dasha significance right
    now. That's the intended "higher priority" effect.
  - If the same disease is produced by more than one planet/tier, its
    score is the BEST (highest) of those paths; every contributing source
    is still kept for the backend-trail view.
  - If neither the current Mahadasha nor Antardasha planet occupies any of
    the 5 hierarchy tiers for this chart, every disease gets +0 and the
    ranking degrades gracefully to plain tier-weight order -- nothing
    breaks, dasha just had no leverage to add for this particular chart.
"""

from priority_cascade import build_cascade_tiers
from knowledge_base import is_term_allowed_for_gender, normalize_gender
from rashi_truth_review import NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS

MAHADASHA_MARKS = 12.0
ANTARDASHA_MARKS = 7.0


def _disease_organs(disease: str, gender: str | None) -> list[str]:
    return [o for o in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []) if is_term_allowed_for_gender(o, gender)]


def compute_dasha_marks(dasha: dict) -> dict:
    """planet -> {"marks": float, "roles": [str, ...]}, only for planets
    that are currently running as Mahadasha and/or Antardasha."""
    dasha = dasha or {}
    maha_planet = ((dasha.get("mahadasha") or {}).get("planet") or "").strip()
    antar_planet = ((dasha.get("antardasha") or {}).get("planet") or "").strip()

    result: dict[str, dict] = {}
    if maha_planet:
        result.setdefault(maha_planet, {"marks": 0.0, "roles": []})
        result[maha_planet]["marks"] += MAHADASHA_MARKS
        result[maha_planet]["roles"].append("Mahadasha")
    if antar_planet:
        result.setdefault(antar_planet, {"marks": 0.0, "roles": []})
        result[antar_planet]["marks"] += ANTARDASHA_MARKS
        result[antar_planet]["roles"].append("Antardasha")
    return result


def run_dasha_weighted_disease_priority(chart_data: dict, rule1_result: dict, dasha: dict, gender: str | None = None) -> dict:
    gender = normalize_gender(gender or chart_data.get("gender"))
    tiers = build_cascade_tiers(chart_data, rule1_result, gender=gender)
    dasha_marks = compute_dasha_marks(dasha)

    if not tiers:
        return {
            "title": "Dasha-Weighted Disease Priority",
            "subtitle": "No cascade tiers resolved for this chart.",
            "dasha_marks": dasha_marks,
            "ranked_diseases": [],
        }

    disease_index: dict[str, dict] = {}
    for tier in tiers:
        tier_weight = tier["weight"]
        for planet in tier.get("planets", []):
            planet_name = planet.get("planet", "")
            marks_info = dasha_marks.get(planet_name, {"marks": 0.0, "roles": []})
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
                    "dasha_roles": marks_info["roles"],
                    "dasha_marks": marks_info["marks"],
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
        "title": "Dasha-Weighted Disease Priority",
        "subtitle": (
            "Independent ranking: cascade tier position + a boost for diseases produced "
            "by the currently running Mahadasha/Antardasha planet. Does not affect the "
            "anchored Disease-Driven Priority Logic or the Shadbala-weighted ranking above."
        ),
        "dasha_marks": dasha_marks,
        "ranked_diseases": results,
    }
