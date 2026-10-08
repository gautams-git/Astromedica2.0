from __future__ import annotations

"""
Anchored Disease-Driven Priority Cascade Logic (v2).

Client correction to v1 (disease_priority_cascade_logic.py):
  v1 clustered diseases across the WHOLE chart by "primary organ" (just the
  first organ listed for a disease in the disease->organ knowledge base).
  That let a disease from a LOW tier hijack a cluster's rank whenever ANY
  organ it merely lists -- not its primary one -- happened to also be listed
  by a Rank-1 disease. Real example that exposed the bug:
    - "Hip diseases" (Rank-1 tier) lists Blood as one of many organs, but
      its primary organ is "Hips Joints". Because v1 let that Blood mention
      drag the whole "Blood" organ-cluster's rank down to 1, "Cuts"/
      "Weakness"/"Piles, Fistula" (all Rank-2+ diseases whose PRIMARY organ
      is Blood) out-ranked "Hip diseases" itself -- even though none of them
      actually share Hip diseases' real organs (Hips Joints/Hips/Thighs/
      Waist/Loins).
    - Meanwhile "Rheumatism" (Rank-1) and "Tumours in the knees" (Rank-2)
      genuinely share Joints/Bones/Knees -- an obvious human-eye match --
      but v1 missed it because their PRIMARY organs ("Joints" vs "Knees")
      differ.

v2 rule (client spec, confirmed 2026-07-01):
  - Rank-1 tier's diseases are the fixed "head". They are never displaced
    by a lower-tier disease, and every one of them is always shown.
  - For each Rank-1 disease, walk the lower tiers in ascending rank order
    (rank 2, then 3, then 4, ...). At each tier, compare that Rank-1
    disease's FULL organ set (every organ its own disease->organ mapping
    lists, not just the first/"primary" one) against the FULL organ set of
    every disease that tier produced. ANY overlap is a match -- this is the
    "think like a human" check, not an exact disease-name match and not a
    single-primary-organ match.
  - A Rank-1 disease that finds a match anywhere below it is "confirmed"
    and is shown ABOVE Rank-1 diseases that found no match at all -- both
    are still Rank-1 by hierarchy, but confirmation earns the top slot
    (this is why "Hip diseases", confirmed by "Piles, Fistula" via Hips/
    Blood/Anus/Bowels/Excretory Organs, must outrank unconfirmed
    "Rheumatism"/"Locomotor ataxia"/"Pulmonary troubles" from the same
    tier).
  - A lower-tier disease can confirm more than one Rank-1 disease if it
    genuinely organ-overlaps more than one -- no reason to hide a real
    match just because it was "claimed" once already.
  - Lower-tier diseases that never confirm any Rank-1 disease are not
    thrown away -- they are listed after the full Rank-1 head, in their
    own tier order, so nothing produced by the chart silently disappears.
  - No weights, no scores: every comparison below is a rank-position
    lookup or a plain count, same constraint as v1.

  - Guard against "generic organ" false matches (client correction,
    confirmed 2026-07-01): a first pass that counted ANY shared organ as a
    match resurfaced the same problem v2 was built to fix -- e.g. "Hip
    diseases" matching "Pimples"/"Cuts"/"Weakness" on "Blood" alone, which
    is not a meaningful human-eye match, just a very common word. "Blood"
    alone appears in 51% of every disease entry in the knowledge base --
    by far the most common organ (next highest is ~19%) -- so a shared
    "Blood" (or any organ at/above GENERIC_ORGAN_FREQUENCY_THRESHOLD) is
    weak evidence on its own. A match is only real when BOTH of these hold:
      1) at least MIN_SHARED_ORGANS organs are shared (a single shared
         organ is never enough), AND
      2) at least one of the shared organs is NOT generic (so two diseases
         that only share generic organs, e.g. Blood + Bones, still don't
         count -- there must be a genuinely specific, distinctive organ in
         common, like "Hips" or "Knees").

Output schema is intentionally identical to v1's ("ranked_diseases" with
rank/disease/disease_organs/related_diseases/sourced_from/agreement_count/
drishti_confirmed/...") so the existing priority6 page can render it with
no template changes.
"""

from knowledge_base import is_term_allowed_for_gender, normalize_gender
from rashi_truth_review import NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS
from priority_cascade import build_cascade_tiers, build_drishti_catalysts

# A shared organ only counts as real evidence of a match if it's not one of
# the handful of organs so common across the knowledge base that sharing it
# proves nothing (e.g. "Blood" alone appears on 51% of all diseases).
GENERIC_ORGAN_FREQUENCY_THRESHOLD = 0.15
MIN_SHARED_ORGANS = 3

# Terms that live in the nakshatra "disease" lists but are procedures/
# outcomes, not diseases -- they must never be treated as evidence that
# confirms a Rank-1 disease (client correction, 2026-07-01: "Amputation"
# is a procedure, not a disease, so it shouldn't match anything).
NON_DISEASE_TERMS = {"amputation"}


def _norm(value: str) -> str:
    return str(value or "").strip().lower()


def _compute_generic_organs() -> set[str]:
    disease_count = len(NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS)
    if not disease_count:
        return set()
    organ_disease_counts: dict[str, int] = {}
    for organs in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.values():
        for organ in {_norm(o) for o in organs}:
            organ_disease_counts[organ] = organ_disease_counts.get(organ, 0) + 1
    return {
        organ
        for organ, count in organ_disease_counts.items()
        if (count / disease_count) >= GENERIC_ORGAN_FREQUENCY_THRESHOLD
    }


GENERIC_ORGANS = _compute_generic_organs()


def _is_real_match(shared_organs: set[str]) -> bool:
    """A match needs >=MIN_SHARED_ORGANS shared organs, AND at least one of
    them must not be generic -- two diseases that only share generic organs
    (e.g. Blood + Bones) are not treated as related."""
    if len(shared_organs) < MIN_SHARED_ORGANS:
        return False
    return bool(shared_organs - GENERIC_ORGANS)


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


def _disease_organs(disease: str, gender: str | None) -> list[str]:
    return [o for o in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []) if is_term_allowed_for_gender(o, gender)]


def _tier_diseases(tier: dict) -> list[str]:
    """All distinct nakshatra diseases this tier's planet(s) produced, in
    first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for planet in tier.get("planets", []):
        for disease in planet.get("nakshatra_diseases", []):
            key = _norm(disease)
            if key and key not in seen:
                seen.add(key)
                out.append(disease)
    return out


def _tier_disease_sources(tier: dict) -> dict[str, list[str]]:
    """disease -> which planet(s) *within this tier* produced it."""
    sources: dict[str, list[str]] = {}
    for planet in tier.get("planets", []):
        planet_name = planet.get("planet", "")
        for disease in planet.get("nakshatra_diseases", []):
            bucket = sources.setdefault(disease, [])
            if planet_name and planet_name not in bucket:
                bucket.append(planet_name)
    return sources


def _tier_planet_organs(tier: dict) -> list[str]:
    from disease_priority_cascade_logic import _canonical_organs  # reuse existing helper
    organs: list[str] = []
    for planet in tier.get("planets", []):
        organs.extend(_canonical_organs(planet.get("planet_terms", [])))
    return _dedupe(organs)


def _tier_rashi_organs(tier: dict) -> list[str]:
    organs: list[str] = []
    for planet in tier.get("planets", []):
        organs.extend(planet.get("rashi_organs", []))
    return _dedupe(organs)


def _terms_of(entity: dict) -> set[str]:
    return {
        _norm(t)
        for t in (
            entity.get("planet_terms", [])
            + entity.get("nakshatra_diseases", [])
            + entity.get("rashi_organs", [])
        )
    }


def run_disease_priority_cascade_logic_v2(chart_data: dict, rule1_result: dict, gender: str | None = None) -> dict:
    gender = normalize_gender(gender or chart_data.get("gender"))

    tiers = build_cascade_tiers(chart_data, rule1_result, gender=gender)
    catalysts = build_drishti_catalysts(chart_data, rule1_result, gender=gender)

    subtitle = (
        "Anchored logic: Rank-1 tier's diseases are the fixed head. Each is checked "
        "against every lower tier's diseases for real organ overlap (not just a shared "
        "primary organ) -- confirmed Rank-1 diseases are shown first, unconfirmed ones "
        "after, then leftover lower-tier diseases in tier order."
    )

    if not tiers:
        return {
            "title": "Disease-Driven Priority Cascade Logic (Anchored)",
            "subtitle": subtitle,
            "cascade_tiers": [],
            "ranked_diseases": [],
        }

    head_tier, *lower_tiers = tiers
    head_diseases = _tier_diseases(head_tier)
    head_sources = _tier_disease_sources(head_tier)

    organs_cache: dict[str, set[str]] = {}

    def organs_of(disease: str) -> set[str]:
        if disease not in organs_cache:
            organs_cache[disease] = {_norm(o) for o in _disease_organs(disease, gender)}
        return organs_cache[disease]

    # Flatten every lower tier's diseases, tagged with their tier, in
    # ascending tier-rank order -- this is the order each Rank-1 disease
    # walks down through when searching for a confirming match.
    lower_tier_diseases: list[tuple[dict, str]] = []
    for tier in lower_tiers:
        for disease in _tier_diseases(tier):
            lower_tier_diseases.append((tier, disease))

    claimed: set[tuple[int, str]] = set()  # (tier_rank, disease_norm) already used as a confirmation

    anchors = []
    for disease in head_diseases:
        if _norm(disease) in NON_DISEASE_TERMS:
            continue
        d_organs = organs_of(disease)
        # Keyed by disease name -- the SAME disease can independently show up
        # in more than one lower tier (e.g. two different planets each list
        # "Diabetes, Insulin deficiency" in their own nakshatra). That's one
        # real-world disease, so it must be one merged match, not a
        # duplicate card repeating the same name twice.
        match_map: dict[str, dict] = {}
        if d_organs:
            for tier, cand_disease in lower_tier_diseases:
                if _norm(cand_disease) == _norm(disease):
                    continue
                if _norm(cand_disease) in NON_DISEASE_TERMS:
                    continue
                shared = d_organs & organs_of(cand_disease)
                if not _is_real_match(shared):
                    continue
                claimed.add((tier["rank"], _norm(cand_disease)))
                key = _norm(cand_disease)
                entry = match_map.setdefault(key, {
                    "disease": cand_disease,
                    "disease_organs": _disease_organs(cand_disease, gender),
                    "matched_organs": set(),
                    "tiers": [],
                    "sourced_from_planets": [],
                })
                entry["matched_organs"] |= shared
                entry["tiers"].append({"rank": tier["rank"], "label": tier["label"]})
                for planet_name in _tier_disease_sources(tier).get(cand_disease, []):
                    if planet_name not in entry["sourced_from_planets"]:
                        entry["sourced_from_planets"].append(planet_name)

        matches = []
        for entry in match_map.values():
            best_tier = min(entry["tiers"], key=lambda t: t["rank"])
            matches.append({
                "disease": entry["disease"],
                "disease_organs": entry["disease_organs"],
                "matched_organs": sorted(entry["matched_organs"]),
                "tier_rank": best_tier["rank"],
                "tier_label": best_tier["label"],
                "tiers": sorted(entry["tiers"], key=lambda t: t["rank"]),
                "sourced_from_planets": entry["sourced_from_planets"],
            })
        matches.sort(key=lambda m: m["tier_rank"])
        anchors.append({
            "disease": disease,
            "disease_organs": _disease_organs(disease, gender),
            "sourced_from_planets": head_sources.get(disease, []),
            "confirmed": bool(matches),
            "matches": matches,
        })

    # Confirmed Rank-1 diseases first, ranked by the STRENGTH of their best
    # single match (highest shared-organ count) -- not by how many diseases
    # confirmed them. A disease confirmed once with 5 shared organs (e.g.
    # Hip diseases <-> Piles, Fistula) is stronger, more specific evidence
    # than one confirmed 3 times with only 4 shared organs each, so match
    # count alone is the wrong signal (client correction, 2026-07-01).
    # Ties fall back to number of confirming matches, then Python's stable
    # sort keeps the nakshatra's original disease order.
    def _best_match_strength(anchor: dict) -> int:
        return max((len(m["matched_organs"]) for m in anchor["matches"]), default=0)

    anchors.sort(key=lambda a: (0 if a["confirmed"] else 1, -_best_match_strength(a), -len(a["matches"])))

    # Leftover lower-tier diseases that never confirmed any Rank-1 disease --
    # grouped by disease name (a disease repeated across tiers keeps every
    # tier it came from), ordered by the earliest tier rank it appears at.
    trailing_map: dict[str, dict] = {}
    for tier, disease in lower_tier_diseases:
        if _norm(disease) in NON_DISEASE_TERMS:
            continue
        if (tier["rank"], _norm(disease)) in claimed:
            continue
        entry = trailing_map.setdefault(_norm(disease), {
            "disease": disease,
            "disease_organs": _disease_organs(disease, gender),
            "tier_ranks": [],
            "tiers": [],
        })
        entry["tier_ranks"].append(tier["rank"])
        entry["tiers"].append(tier)

    trailing = sorted(trailing_map.values(), key=lambda e: min(e["tier_ranks"]))

    def _drishti_confirmations_for(contributing_tiers: list[dict]) -> list[dict]:
        confirmations = []
        for tier in contributing_tiers:
            tier_terms: set[str] = set()
            for planet in tier.get("planets", []):
                tier_terms |= _terms_of(planet)
            for catalyst in catalysts:
                shared = tier_terms & _terms_of(catalyst)
                if shared:
                    confirmations.append({
                        "planet": catalyst.get("planet", ""),
                        "aspect_type": catalyst.get("aspect_type", "aspect"),
                        "confirmed_via_tier": tier["label"],
                        "shared_terms": sorted(shared),
                    })
        return confirmations

    results = []
    rank = 1

    for anchor in anchors:
        confirming_tiers: dict[int, str] = {}
        for m in anchor["matches"]:
            for t in m["tiers"]:
                confirming_tiers[t["rank"]] = t["label"]
        contributing_tiers = [head_tier] + [t for t in lower_tiers if t["rank"] in confirming_tiers]

        sourced_from = [{"rank": head_tier["rank"], "label": head_tier["label"]}]
        sourced_from += [{"rank": r, "label": label} for r, label in sorted(confirming_tiers.items())]

        supporting_planet_organs = _dedupe([o for t in contributing_tiers for o in _tier_planet_organs(t)])
        supporting_rashi_organs = _dedupe([o for t in contributing_tiers for o in _tier_rashi_organs(t)])

        drishti_confirmations = _drishti_confirmations_for(contributing_tiers)

        results.append({
            "rank": rank,
            "disease": anchor["disease"],
            "disease_organs": anchor["disease_organs"],
            "confirmed": anchor["confirmed"],
            "is_anchor": True,
            "related_diseases": [
                {
                    "disease": m["disease"],
                    "disease_organs": m["disease_organs"],
                    "matched_organs": m["matched_organs"],
                    "sourced_from": [{"rank": t["rank"], "label": t["label"]} for t in m["tiers"]],
                }
                for m in anchor["matches"]
            ],
            "sourced_from": sourced_from,
            "agreement_count": 1 + len(confirming_tiers),
            "drishti_confirmed": bool(drishti_confirmations),
            "drishti_confirmations": drishti_confirmations,
            "supporting_planet_organs": supporting_planet_organs,
            "supporting_rashi_organs": supporting_rashi_organs,
        })
        rank += 1

    for entry in trailing:
        contributing_tiers = entry["tiers"]
        sourced_from = [{"rank": t["rank"], "label": t["label"]} for t in contributing_tiers]
        supporting_planet_organs = _dedupe([o for t in contributing_tiers for o in _tier_planet_organs(t)])
        supporting_rashi_organs = _dedupe([o for t in contributing_tiers for o in _tier_rashi_organs(t)])
        drishti_confirmations = _drishti_confirmations_for(contributing_tiers)

        results.append({
            "rank": rank,
            "disease": entry["disease"],
            "disease_organs": entry["disease_organs"],
            "confirmed": False,
            "is_anchor": False,
            "related_diseases": [],
            "sourced_from": sourced_from,
            "agreement_count": len(set(entry["tier_ranks"])),
            "drishti_confirmed": bool(drishti_confirmations),
            "drishti_confirmations": drishti_confirmations,
            "supporting_planet_organs": supporting_planet_organs,
            "supporting_rashi_organs": supporting_rashi_organs,
        })
        rank += 1

    return {
        "title": "Disease-Driven Priority Cascade Logic (Anchored)",
        "subtitle": subtitle,
        "cascade_tiers": [
            {"rank": t["rank"], "tier_id": t["tier_id"], "label": t["label"], "planets": [p.get("planet", "") for p in t.get("planets", [])]}
            for t in tiers
        ],
        "ranked_diseases": results,
    }


def build_organ_priority_view(disease_priority_result: dict) -> dict:
    """
    Organ-Driven Priority view, derived directly from the disease ranking
    above -- NOT a second, independent matching algorithm. This avoids two
    cascades ever disagreeing with each other.

    An organ's rank is the best (lowest) disease-rank it appears at. Within
    that, an organ that was itself part of a real cross-tier MATCH (i.e. it's
    one of a disease's matched_organs against a confirmed related disease) is
    ranked above an organ that only ever showed up as a disease's own
    supporting organ list -- same "confirmed beats standalone" rule already
    used for diseases. No weights/scores: rank position + plain counts only.
    """
    ranked_diseases = disease_priority_result.get("ranked_diseases", [])

    organ_index: dict[str, dict] = {}
    for d in ranked_diseases:
        disease_rank = d["rank"]
        confirmed_organs = {
            _norm(o)
            for r in d.get("related_diseases", [])
            for o in r.get("matched_organs", [])
        }
        for organ in d.get("disease_organs", []):
            key = _norm(organ)
            is_confirmed_here = key in confirmed_organs
            entry = organ_index.setdefault(key, {
                "organ": organ,
                "best_rank": disease_rank,
                "confirmed": False,
                "confirming_diseases": set(),
                "appeared_in_diseases": [],
            })
            entry["best_rank"] = min(entry["best_rank"], disease_rank)
            entry["appeared_in_diseases"].append({
                "disease": d["disease"],
                "rank": disease_rank,
                "confirmed": is_confirmed_here,
            })
            if is_confirmed_here:
                entry["confirmed"] = True
                entry["confirming_diseases"].add(d["disease"])

    organs = list(organ_index.values())
    organs.sort(key=lambda o: (
        o["best_rank"],
        0 if o["confirmed"] else 1,
        -len(o["confirming_diseases"]),
        o["organ"],
    ))

    results = []
    for rank, o in enumerate(organs, start=1):
        results.append({
            "rank": rank,
            "organ": o["organ"],
            "confirmed": o["confirmed"],
            "best_disease_rank": o["best_rank"],
            "confirming_diseases": sorted(o["confirming_diseases"]),
            "appeared_in_diseases": sorted(o["appeared_in_diseases"], key=lambda a: a["rank"]),
        })

    return {
        "title": "Organ-Driven Priority (derived from Disease-Driven Priority Logic)",
        "subtitle": (
            "Organs ranked by the best disease rank they appear in above -- not a "
            "second algorithm. An organ confirmed by a real cross-tier disease match "
            "ranks above one that only ever showed up as supporting context."
        ),
        "ranked_organs": results,
    }
