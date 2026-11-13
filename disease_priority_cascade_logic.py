from __future__ import annotations

"""
Disease-Driven Priority Cascade Logic.

Client spec:
  - Diseases drive the ranking. NO weights, NO scores, NO summed/multiplied
    numbers anywhere in this file -- every comparison below is either a
    rank-position lookup or a plain count (frequency), never arithmetic
    combined into a score.
  - Ranking walks the same priority hierarchy used everywhere else:
        6th House Occupant -> 11th House Occupant -> 6th House Lord ->
        Dispositor of 6th Lord -> Asc/Moon (if related)
    (with the usual compression: if 6th Occupant is absent, 11th Occupant
    becomes rank 1; if both are absent, 6th Lord becomes rank 1.)

  - "Think like a human" upgrade: agreement is detected at the ORGAN level,
    not just the exact-disease-string level. If 6th House Occupant produces
    a disease that maps to Kidney, and 11th House Occupant separately
    produces a *different* disease that also maps to Kidney, that is
    cross-tier agreement on Kidney -- even though the two disease names
    don't match. Kidney-related diseases should rank above an organ only
    confirmed by one tier.

  - Diseases are grouped into organ clusters. Each disease's "primary organ"
    is simply the first organ listed for it in the disease->organ knowledge
    base (a stable, deterministic choice -- not a judgment call made here).
    Within a cluster, one disease becomes the headline; the rest are listed
    underneath as related diseases.

  - Headline selection within a cluster uses BOTH signals the client asked
    for, in this order (still ordinal, never summed):
      1) The disease's own best (lowest) tier rank.
      2) The disease's own frequency -- how many tiers independently
         produced that exact disease.
      3) A positional organ-overlap tiebreak (same as before).
      4) Alphabetical, as a final stable fallback.

  - Cluster (organ) ranking uses the same two keys, one level up:
      1) The best tier rank of ANY disease in the cluster.
      2) How many DISTINCT tiers point to that organ at all (frequency of
         the organ across the hierarchy, not the disease).

  - Drishti (aspects) remains a binary confirmation tag only -- it is
    computed AFTER ranking and never changes a disease's or cluster's
    position.
  - Planet Organs and Rashi Organs are shown AFTER the headline disease, as
    supporting context -- never as scoring inputs.

This file is intentionally isolated from every other engine (v1, v2, or
the shared priority_cascade.py's weight fields) so it can be revised
independently later.
"""

from knowledge_base import is_term_allowed_for_gender, normalize_gender
from rashi_truth_review import ALL_ORGANS, NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS
from priority_cascade import build_cascade_tiers, build_drishti_catalysts

ALL_ORGANS_LOOKUP = {str(item).strip().lower(): item for item in ALL_ORGANS}


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


def _canonical_organs(terms: list[str]) -> list[str]:
    organs = []
    for term in terms:
        canonical = ALL_ORGANS_LOOKUP.get(_norm(term))
        if canonical:
            organs.append(canonical)
    return _dedupe(organs)


def _tier_planet_organs(tier: dict) -> list[str]:
    organs: list[str] = []
    for planet in tier.get("planets", []):
        organs.extend(_canonical_organs(planet.get("planet_terms", [])))
    return _dedupe(organs)


def _tier_rashi_organs(tier: dict) -> list[str]:
    organs: list[str] = []
    for planet in tier.get("planets", []):
        organs.extend(planet.get("rashi_organs", []))
    return _dedupe(organs)


def _disease_organs(disease: str, gender: str | None) -> list[str]:
    return [o for o in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []) if is_term_allowed_for_gender(o, gender)]


def run_disease_priority_cascade_logic(chart_data: dict, rule1_result: dict, gender: str | None = None) -> dict:
    gender = normalize_gender(gender or chart_data.get("gender"))

    tiers = build_cascade_tiers(chart_data, rule1_result, gender=gender)
    catalysts = build_drishti_catalysts(chart_data, rule1_result, gender=gender)

    # -- Step 1: index every disease by which tiers produced it -----------
    # disease -> {"first_rank", "tier_ranks": set, "sourced_from": [...]}
    disease_index: dict[str, dict] = {}

    for tier in tiers:
        tier_diseases: set[str] = set()
        for planet in tier.get("planets", []):
            tier_diseases.update(planet.get("nakshatra_diseases", []))
        if not tier_diseases:
            continue
        for disease in tier_diseases:
            entry = disease_index.setdefault(disease, {
                "first_rank": tier["rank"],
                "tier_ranks": set(),
                "sourced_from": [],
            })
            entry["first_rank"] = min(entry["first_rank"], tier["rank"])
            entry["tier_ranks"].add(tier["rank"])
            entry["sourced_from"].append({
                "rank": tier["rank"],
                "label": tier["label"],
                "planets": [p.get("planet", "") for p in tier.get("planets", [])],
            })

    disease_organs_cache = {d: _disease_organs(d, gender) for d in disease_index}

    def _organ_tiebreak_rank(disease: str) -> int:
        """Walk the hierarchy in order; return the rank of the FIRST tier
        whose Planet/Rashi Organs overlap this disease's own organs."""
        disease_organs = {_norm(o) for o in disease_organs_cache.get(disease, [])}
        if not disease_organs:
            return 9999
        for tier in tiers:
            tier_organs = {_norm(o) for o in (_tier_planet_organs(tier) + _tier_rashi_organs(tier))}
            if disease_organs & tier_organs:
                return tier["rank"]
        return 9999

    def _disease_sort_key(disease: str) -> tuple:
        entry = disease_index[disease]
        return (
            entry["first_rank"],            # 1) earliest tier wins
            -len(entry["tier_ranks"]),       # 2) frequency: tiers agreeing on this exact disease
            _organ_tiebreak_rank(disease),    # 3) positional organ-overlap tiebreak
            disease,                          # 4) alphabetical, final fallback
        )

    # -- Step 2: build organ-level agreement, and assign each disease to --
    # -- its primary organ (the first organ in its own mapping) -----------
    # organ -> {"tier_ranks": set, "first_rank": int, "cluster_diseases": [disease,...]}
    organ_index: dict[str, dict] = {}
    unclustered_diseases: list[str] = []

    for disease, entry in disease_index.items():
        organs = disease_organs_cache.get(disease, [])
        if not organs:
            unclustered_diseases.append(disease)
            continue
        for organ in organs:
            oentry = organ_index.setdefault(organ, {
                "tier_ranks": set(),
                "first_rank": entry["first_rank"],
                "cluster_diseases": [],
            })
            oentry["tier_ranks"] |= entry["tier_ranks"]
            oentry["first_rank"] = min(oentry["first_rank"], entry["first_rank"])
        primary_organ = organs[0]
        organ_index[primary_organ]["cluster_diseases"].append(disease)

    # -- Step 3: rank clusters (organ-level), then pick a headline disease
    cluster_keys: list[tuple] = []  # (sort_key, kind, organ_or_none, diseases_in_cluster, tier_ranks_for_display)

    for organ, data in organ_index.items():
        if not data["cluster_diseases"]:
            continue
        sort_key = (
            data["first_rank"],
            -len(data["tier_ranks"]),
        )
        cluster_keys.append((sort_key, "organ", organ, data["cluster_diseases"], data["tier_ranks"]))

    for disease in unclustered_diseases:
        entry = disease_index[disease]
        sort_key = (entry["first_rank"], -len(entry["tier_ranks"]))
        cluster_keys.append((sort_key, "single", None, [disease], entry["tier_ranks"]))

    cluster_keys.sort(key=lambda item: (item[0], item[2] or item[3][0]))

    # -- Step 4: build the final output ------------------------------------
    def _drishti_confirmations_for(contributing_tiers: list[dict]) -> list[dict]:
        confirmations = []
        for tier in contributing_tiers:
            tier_terms = {
                _norm(t)
                for planet in tier.get("planets", [])
                for t in (planet.get("planet_terms", []) + planet.get("nakshatra_diseases", []) + planet.get("rashi_organs", []))
            }
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
                    confirmations.append({
                        "planet": catalyst.get("planet", ""),
                        "aspect_type": catalyst.get("aspect_type", "aspect"),
                        "confirmed_via_tier": tier["label"],
                        "shared_terms": sorted(shared),
                    })
        return confirmations

    results = []
    for rank, (_sort_key, kind, organ, diseases, cluster_tier_ranks) in enumerate(cluster_keys, start=1):
        ordered_diseases = sorted(diseases, key=_disease_sort_key)
        headline = ordered_diseases[0]
        related = ordered_diseases[1:]

        headline_entry = disease_index[headline]
        contributing_tiers = [tier for tier in tiers if tier["rank"] in headline_entry["tier_ranks"]]

        # Organ-level agreement: every tier touching ANY disease that maps to
        # this organ -- precomputed in organ_index, including diseases whose
        # PRIMARY organ is different but still list this organ secondarily.
        cluster_tiers = [tier for tier in tiers if tier["rank"] in cluster_tier_ranks]

        supporting_planet_organs = _dedupe([o for tier in cluster_tiers for o in _tier_planet_organs(tier)])
        supporting_rashi_organs = _dedupe([o for tier in cluster_tiers for o in _tier_rashi_organs(tier)])

        drishti_confirmations = _drishti_confirmations_for(contributing_tiers)

        results.append({
            "rank": rank,
            "organ": organ,
            "disease": headline,
            "related_diseases": [
                {
                    "disease": d,
                    "sourced_from": sorted(disease_index[d]["sourced_from"], key=lambda s: s["rank"]),
                    "disease_organs": disease_organs_cache.get(d, []),
                }
                for d in related
            ],
            "sourced_from": sorted(headline_entry["sourced_from"], key=lambda s: s["rank"]),
            "agreement_count": len(headline_entry["tier_ranks"]),
            "organ_agreement_count": len(cluster_tier_ranks) if organ else len(headline_entry["tier_ranks"]),
            "drishti_confirmed": bool(drishti_confirmations),
            "drishti_confirmations": drishti_confirmations,
            "disease_organs": disease_organs_cache.get(headline, []),
            "supporting_planet_organs": supporting_planet_organs,
            "supporting_rashi_organs": supporting_rashi_organs,
        })

    return {
        "title": "Disease-Driven Priority Cascade Logic",
        "subtitle": "Disease-based logic: ranked by priority hierarchy and repetition across sources -- no weights or scores. Planet/Rashi organs shown as supporting context after each disease.",
        "cascade_tiers": [
            {"rank": t["rank"], "tier_id": t["tier_id"], "label": t["label"], "planets": [p.get("planet", "") for p in t.get("planets", [])]}
            for t in tiers
        ],
        "ranked_diseases": results,
    }
