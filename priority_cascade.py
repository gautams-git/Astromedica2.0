from __future__ import annotations

"""
Shared priority-cascade resolver for the new (v2) algorithm set.

Cascade rule (client spec):
  6th House Occupant and 11th House Occupant are both "main" priorities.
  - If 6th House Occupant is present: order is
        6th Occupant -> 11th Occupant -> 6th Lord -> 6th Dispositor -> Asc/Moon (if related)
  - If 6th House Occupant is absent but 11th House Occupant is present:
        11th Occupant -> 6th Lord -> 6th Dispositor -> Asc/Moon (if related)
  - If both 6th and 11th House Occupants are absent:
        6th Lord -> 6th Dispositor -> Asc/Moon (if related)

Whichever tiers are actually present get compressed into ranks 1..N with no
gaps -- there is never a "fallback label", the cascade just naturally drops
empty tiers.

Drishti (aspects) is NOT a tier and never contributes its own diseases/organs.
It only multiplies the score of a tier it aspects, when the aspecting planet
shares a disease/organ with that tier's planet(s).
"""

from knowledge_base import (
    NAKSHATRA_DISEASES,
    PLANET_DISEASES,
    RASHI_ORGANS,
    is_term_allowed_for_gender,
)
from rashi_truth_review import NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS

# Weight assigned by *cascade rank position*, not by fixed P-label.
# Reused from the existing, already-tuned decay curve (disease_first_logic.py).
RANK_WEIGHT_SCHEDULE = [6.0, 4.5, 3.5, 2.8, 2.0]
RANK_WEIGHT_FLOOR = 1.5  # any tier beyond the schedule length

PLANET_LAYER_RATIO = 0.5
RASHI_LAYER_RATIO = 0.5

# Drishti catalyst: +20% per matching aspecting planet, capped at +40% total.
CATALYST_PER_MATCH = 0.2
CATALYST_CAP = 0.4

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


def term_allowed(term: str, gender: str | None) -> bool:
    return is_term_allowed_for_gender(term, gender)


def planet_terms(planet_name: str) -> list[str]:
    raw = PLANET_DISEASES.get(planet_name, "")
    terms: list[str] = []
    for term in str(raw).split(","):
        clean = term.strip()
        if not clean or any(clean.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        terms.append(clean)
    return terms


def nakshatra_terms(nakshatra_name: str, pada) -> list[str]:
    nk_data = NAKSHATRA_DISEASES.get(nakshatra_name, {})
    if not nk_data:
        return []
    pada_str = str(pada)
    for key, diseases in nk_data.items():
        padas_in_key = [p.strip() for p in str(key).split(",")]
        if pada_str in padas_in_key:
            return list(diseases)
    return []


def rashi_terms(rashi_name: str) -> list[str]:
    raw = RASHI_ORGANS.get(rashi_name, "")
    return [term.strip() for term in str(raw).split(",") if term.strip()]


def _planet_data(chart_data: dict, planet_name: str) -> dict:
    return next(
        (p for p in chart_data.get("planets", []) if p.get("name") == planet_name),
        {},
    ) or {}


def _disease_organ_breakdown(diseases: list[str], gender: str | None) -> list[dict]:
    """For each nakshatra disease, the organs that specific disease maps to
    (via the disease->organ knowledge base) -- one entry per disease, shown
    stacked in the UI as 'Disease -> Organ, Organ'."""
    breakdown = []
    for disease in diseases:
        organs = [o for o in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []) if term_allowed(o, gender)]
        if organs:
            breakdown.append({"disease": disease, "organs": _dedupe(organs)})
    return breakdown


def _planet_snapshot(chart_data: dict, planet_name: str, gender: str | None) -> dict:
    """Diseases/organs for a single planet at its current placement."""
    planet_data = _planet_data(chart_data, planet_name)
    if not planet_data:
        return {
            "planet": planet_name,
            "sign": "",
            "nakshatra": "",
            "pada": 1,
            "planet_terms": [],
            "nakshatra_diseases": [],
            "disease_organ_breakdown": [],
            "rashi_organs": [],
        }

    sign_name = (
        (planet_data.get("sign") or {}).get("name", "")
        if isinstance(planet_data.get("sign"), dict)
        else planet_data.get("sign", "")
    )
    nk = planet_data.get("nakshatra", {}) or {}
    nk_name = nk.get("name", "") if isinstance(nk, dict) else ""
    nk_pada = nk.get("pada", 1) if isinstance(nk, dict) else planet_data.get("nakshatra_pada", 1)

    p_terms = [t for t in planet_terms(planet_name) if term_allowed(t, gender)]
    nk_diseases = [d for d in nakshatra_terms(nk_name, nk_pada) if term_allowed(d, gender)]
    r_organs = [o for o in rashi_terms(sign_name) if term_allowed(o, gender)]
    deduped_nk_diseases = _dedupe(nk_diseases)

    return {
        "planet": planet_name,
        "sign": sign_name,
        "nakshatra": nk_name,
        "pada": nk_pada,
        "planet_terms": _dedupe(p_terms),
        "nakshatra_diseases": deduped_nk_diseases,
        "disease_organ_breakdown": _disease_organ_breakdown(deduped_nk_diseases, gender),
        "rashi_organs": _dedupe(r_organs),
    }


def _is_ascendant_lord_related(rule1_result: dict, ascendant_lord: str) -> bool:
    """Asc lord counts as 'related' only if it is itself the 6th occupant,
    6th lord, dispositor of 6th lord, or aspects (drishti) the 6th house --
    i.e. it is already part of the Rule-1 chain, not just structurally present."""
    return _ascendant_lord_relation_role(rule1_result, ascendant_lord) is not None


def _ascendant_lord_relation_role(rule1_result: dict, ascendant_lord: str) -> str | None:
    """Returns the relation role ('occupant'/'6th_house_lord'/'dispositor'/
    'drishti') that makes the Ascendant Lord related, or None if unrelated."""
    if not ascendant_lord:
        return None

    occupant_names = {p.get("name", "") for p in (rule1_result.get("rogkaraka_1") or []) if p.get("name")}
    if ascendant_lord in occupant_names:
        return "occupant"

    sixth_lord = rule1_result.get("sixth_house_lord", "")
    if ascendant_lord and ascendant_lord == sixth_lord:
        return "6th_house_lord"

    dispositor_name = (rule1_result.get("rogkaraka_3") or {}).get("name", "")
    if ascendant_lord and ascendant_lord == dispositor_name:
        return "dispositor"

    drishti_planets = {d.get("planet", "") for d in (rule1_result.get("drishti_on_6th") or []) if d.get("planet")}
    if ascendant_lord in drishti_planets:
        return "drishti"

    return None


def _all_tier_candidates(rule1_result: dict) -> list[dict]:
    """All 5 conceptual cascade slots in fixed semantic order, each carrying
    its planet list (empty if that slot is absent for this chart)."""
    candidates: list[dict] = []

    occupant_list = rule1_result.get("rogkaraka_1") or []
    candidates.append({
        "tier_id": "sixth_occupant",
        "label": "6th House Occupant",
        "planets": [p.get("name", "") for p in occupant_list if p.get("name")],
        "include_rashi": True,
    })

    eleventh_list = rule1_result.get("eleventh_house_occupants") or []
    candidates.append({
        "tier_id": "eleventh_occupant",
        "label": "11th House Occupant",
        "planets": [p.get("name", "") for p in eleventh_list if p.get("name")],
        "include_rashi": True,
    })

    sixth_lord = rule1_result.get("sixth_house_lord", "")
    candidates.append({
        "tier_id": "sixth_lord",
        "label": "6th House Lord",
        "planets": [sixth_lord] if sixth_lord else [],
        "include_rashi": True,
    })

    dispositor_name = (rule1_result.get("rogkaraka_3") or {}).get("name", "")
    candidates.append({
        "tier_id": "sixth_dispositor",
        "label": "Dispositor of 6th Lord",
        "planets": [dispositor_name] if dispositor_name else [],
        "include_rashi": True,
    })

    asc_moon_planets: list[str] = []
    planet_meta: dict[str, dict] = {}
    ascendant_lord = rule1_result.get("ascendant_lord", "")
    asc_role = _ascendant_lord_relation_role(rule1_result, ascendant_lord)
    if asc_role:
        asc_moon_planets.append(ascendant_lord)
        planet_meta[ascendant_lord] = {
            "relation_role": asc_role,
            "relation_reason": f"Ascendant Lord {ascendant_lord} is related via {asc_role.replace('_', ' ')}.",
        }
    conditional_moon = rule1_result.get("conditional_moon") or {}
    if conditional_moon.get("include") and "Moon" not in asc_moon_planets:
        asc_moon_planets.append("Moon")
        planet_meta["Moon"] = {
            "relation_role": conditional_moon.get("connected_role", ""),
            "relation_reason": conditional_moon.get("reason", ""),
        }
    candidates.append({
        "tier_id": "asc_moon",
        "label": "Asc/Moon (Related)",
        "planets": asc_moon_planets,
        "planet_meta": planet_meta,
        "include_rashi": True,
    })

    return candidates


def build_full_cascade_view(rule1_result: dict) -> list[dict]:
    """
    Display-only view of ALL 5 cascade slots, including absent ones (marked
    present=False, rank=None). Used by the UI so 11th House Occupant (etc.)
    is always visible, even when it contributes nothing to scoring.
    """
    candidates = _all_tier_candidates(rule1_result)
    present_candidates = [c for c in candidates if c["planets"]]
    rank_by_tier_id = {c["tier_id"]: rank for rank, c in enumerate(present_candidates, start=1)}

    view = []
    for candidate in candidates:
        rank = rank_by_tier_id.get(candidate["tier_id"])
        present = rank is not None
        weight = None
        if present:
            weight = RANK_WEIGHT_SCHEDULE[rank - 1] if rank <= len(RANK_WEIGHT_SCHEDULE) else RANK_WEIGHT_FLOOR
        planet_meta = candidate.get("planet_meta", {})
        view.append({
            "tier_id": candidate["tier_id"],
            "label": candidate["label"],
            "present": present,
            "rank": rank,
            "weight": round(weight, 2) if weight is not None else None,
            "planets": [
                {
                    "name": name,
                    "relation_role": planet_meta.get(name, {}).get("relation_role", ""),
                    "relation_reason": planet_meta.get(name, {}).get("relation_reason", ""),
                }
                for name in candidate["planets"]
            ],
        })
    return view


def build_cascade_tiers(chart_data: dict, rule1_result: dict, gender: str | None = None) -> list[dict]:
    """
    Returns an ordered list of active cascade tiers, already compressed to
    ranks 1..N (no gaps), each carrying its resolved weight and the
    diseases/organs of every planet in that tier.
    """
    candidates = _all_tier_candidates(rule1_result)
    raw_tiers = [c for c in candidates if c["planets"]]

    tiers: list[dict] = []
    for rank, tier in enumerate(raw_tiers, start=1):
        weight = (
            RANK_WEIGHT_SCHEDULE[rank - 1]
            if rank <= len(RANK_WEIGHT_SCHEDULE)
            else RANK_WEIGHT_FLOOR
        )
        planet_snapshots = [
            _planet_snapshot(chart_data, name, gender)
            for name in tier["planets"]
            if name
        ]
        tiers.append({
            "tier_id": tier["tier_id"],
            "label": tier["label"],
            "rank": rank,
            "weight": round(weight, 2),
            "planet_layer_weight": round(weight * PLANET_LAYER_RATIO, 2),
            "rashi_layer_weight": round(weight * RASHI_LAYER_RATIO, 2),
            "planets": planet_snapshots,
        })

    return tiers


def build_ascendant_reference(chart_data: dict, rule1_result: dict, gender: str | None = None) -> dict:
    """
    Ascendant (sign) and Ascendant Lord (planet) shown unconditionally for
    reference -- independent of whether the Ascendant Lord is "related"
    enough to enter the Asc/Moon cascade tier.
    """
    asc_sign = (chart_data.get("ascendant") or {}).get("name", "")
    asc_organs = [o for o in rashi_terms(asc_sign) if term_allowed(o, gender)]

    ascendant_lord = rule1_result.get("ascendant_lord", "")
    lord_snapshot = _planet_snapshot(chart_data, ascendant_lord, gender) if ascendant_lord else None

    return {
        "ascendant": {"sign": asc_sign, "rashi_organs": _dedupe(asc_organs)},
        "ascendant_lord": lord_snapshot,
    }


def build_house_occupant_reference(chart_data: dict, rule1_result: dict, house_num: int, gender: str | None = None) -> list[dict]:
    """
    All occupants of a specific house (e.g. 2nd house), shown unconditionally
    for reference -- whether or not they're "related" to the Rule-1 chain.
    Reference only; does not feed cascade scoring.
    """
    relations = rule1_result.get("house_occupant_relations") or {}
    house_relations = relations.get(str(house_num)) or relations.get(house_num) or []

    references = []
    for rel in house_relations:
        planet_name = rel.get("name", "")
        if not planet_name:
            continue
        snapshot = _planet_snapshot(chart_data, planet_name, gender)
        references.append({
            **snapshot,
            "related": bool(rel.get("related")),
            "connected_role": rel.get("connected_role", ""),
            "connection_reason": rel.get("connection_reason", ""),
        })
    return references


def build_drishti_catalysts(chart_data: dict, rule1_result: dict, gender: str | None = None) -> list[dict]:
    """Aspecting (drishti) planets on the 6th house, with their own
    diseases/organs -- used only to catalyze matching cascade tiers,
    never as a standalone source."""
    catalysts = []
    for drishti in rule1_result.get("drishti_on_6th") or []:
        planet_name = drishti.get("planet", "")
        if not planet_name:
            continue
        snapshot = _planet_snapshot(chart_data, planet_name, gender)
        catalysts.append({
            **snapshot,
            "aspect_type": drishti.get("aspect_type", "aspect"),
            "from_house": drishti.get("from_house"),
        })
    return catalysts


def _tier_evidence_terms(tier: dict) -> set[str]:
    terms: set[str] = set()
    for planet in tier.get("planets", []):
        terms.update(planet.get("planet_terms", []))
        terms.update(planet.get("nakshatra_diseases", []))
        terms.update(planet.get("rashi_organs", []))
    return {_norm(t) for t in terms}


def catalyst_multiplier(tier: dict, catalysts: list[dict]) -> tuple[float, list[dict]]:
    """
    Returns (multiplier, matched_catalysts) for a given cascade tier.
    Multiplier is 1.0 (no effect) unless an aspecting planet shares a
    disease/organ term with the tier -- then it adds CATALYST_PER_MATCH per
    distinct matching planet, capped at CATALYST_CAP total.
    """
    tier_terms = _tier_evidence_terms(tier)
    if not tier_terms:
        return 1.0, []

    matched: list[dict] = []
    for catalyst in catalysts:
        catalyst_terms = {
            _norm(t)
            for t in (
                catalyst.get("planet_terms", [])
                + catalyst.get("nakshatra_diseases", [])
                + catalyst.get("rashi_organs", [])
            )
        }
        shared = tier_terms & catalyst_terms
        if shared:
            matched.append({
                "planet": catalyst.get("planet", ""),
                "aspect_type": catalyst.get("aspect_type", "aspect"),
                "shared_terms": sorted(shared),
            })

    boost = min(CATALYST_CAP, CATALYST_PER_MATCH * len(matched))
    return round(1.0 + boost, 3), matched
