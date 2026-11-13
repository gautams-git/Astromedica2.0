# hitlist_engine.py 
# ══════════════════════════════════════════════ 
# HITLIST ENGINE — Independent disease priority 
# scoring system. Does NOT use or modify any 
# existing rules.py logic. 
# ══════════════════════════════════════════════ 
 
from term_to_systems import TERM_TO_SYSTEMS, MERGE_GROUPS 
from knowledge_base import ( 
    PLANET_DISEASES, 
    NAKSHATRA_DISEASES, 
    RASHI_ORGANS,
    is_term_allowed_for_gender,
    normalize_gender,
) 
 
# ── Source weights ───────────────────────────── 
SOURCE_WEIGHTS = { 
    "P1": 8.0,   # 6th house occupant (Increased from 3.0)
    "P2": 4.0,   # Dispositor of 6th lord (Increased from 2.0)
    "P3": 2.5,   # Drishti planet (Increased from 1.5)
    "P4": 2.0,   # 6th house lord (Increased from 1.0)
    "P5": 1.5,   # Ascendant lord (Increased from 1.0)
    "P7": 3.0,   # Ascendant rashi (Reduced from 4.0)
    "PM1": 5.0,  # Conditional Moon via occupant
    "PM2": 4.0,  # Conditional Moon via dispositor
    "PM3": 3.5,  # Conditional Moon via drishti
    "PM4": 3.0,  # Conditional Moon via 6th house lord
} 

# ── HitList Layer Multipliers ──────────────────
# HL1 = Planet, HL2 = Nakshatra, HL3 = Rashi
HL_MULTIPLIERS = {
    "HL1": 1.5,  # Planet diseases
    "HL2": 6.0,  # Nakshatra diseases (Increased from 5.0)
    "HL3": 7.0,  # Rashi organs
} 

P1_COHERENCE_MULTIPLIERS = {
    3: 1.0,
    2: 0.8,
    1: 0.5,
    0: 0.5,
}
# ── Convergence multipliers ──────────────────── 
CONVERGENCE = { 
    3: 2.0,   # all 3 HitLists agree 
    2: 1.4,   # 2 HitLists agree 
    1: 1.0,   # only 1 HitList 
} 
 
# skip these from planet disease strings 
SKIP_PREFIXES = [ 
    "Further everything", 
    "Further all" 
] 
 
 
def get_planet_terms(planet_name): 
    """Returns list of disease terms for a planet.""" 
    raw = PLANET_DISEASES.get(planet_name, "") 
    terms = [] 
    for t in raw.split(","): 
        t = t.strip() 
        if not t: 
            continue 
        if any(t.startswith(s) for s in SKIP_PREFIXES): 
            continue 
        terms.append(t) 
    return terms 
 
 
def get_nakshatra_terms(nakshatra_name, pada): 
    """ 
    Returns disease terms for a nakshatra + pada. 
    Matches pada against all pada keys in KB. 
    pada is an integer 1-4. 
    """ 
    nk_data = NAKSHATRA_DISEASES.get(nakshatra_name, {}) 
    if not nk_data: 
        return [] 
    pada_str = str(pada) 
    for key, diseases in nk_data.items(): 
        padas_in_key = [p.strip() for p in key.split(",")] 
        if pada_str in padas_in_key: 
            return diseases 
    return [] 
 
 
def get_rashi_terms(rashi_name): 
    """Returns list of organ terms for a rashi.""" 
    raw = RASHI_ORGANS.get(rashi_name, "") 
    return [t.strip() for t in raw.split(",") if t.strip()] 
 
 
def score_terms(terms, weight, system_scores, hitlist_tracker, hitlist_id): 
    """ 
    For each term, look up its systems in TERM_TO_SYSTEMS. 
    Add weighted score to each system. 
    specificity = 1 / number of systems the term maps to. 
    Track which hitlist contributed to each system. 
    """ 
    for term in terms: 
        systems = TERM_TO_SYSTEMS.get(term, []) 
        if not systems: 
            continue 
        specificity = 1.0 / len(systems) 
        for system in systems: 
            if system not in system_scores: 
                system_scores[system] = 0.0 
            if system not in hitlist_tracker: 
                hitlist_tracker[system] = set() 
            system_scores[system] += weight * specificity 
            hitlist_tracker[system].add(hitlist_id) 
 
 
def is_term_allowed(term, gender):
    return is_term_allowed_for_gender(term, gender)


def build_hitlists(priority_planets, gender=None): 
    """ 
    priority_planets is a list of dicts: 
    [ 
        { 
            "priority": "P1", 
            "planet":   "Saturn", 
            "nakshatra":"Moola", 
            "pada":     3, 
            "rashi":    "Dhanu" 
        }, 
        ... 
    ] 
 
    Returns scored and ranked system groups. 
    """ 
 
    # system_scores[system] = raw score 
    system_scores = {} 
 
    # hitlist_tracker[system] = set of 
    # hitlist ids that contributed 
    # HL1=planet, HL2=nakshatra, HL3=rashi 
    hitlist_tracker = {} 
 
    # Also track which terms contributed 
    # for display/debug purposes 
    term_trail = {} 
    p1_system_scores = {}
    p1_system_layers = {}

    hl1_terms = set()
    hl2_terms = set()
    hl3_terms = set()

    for p in priority_planets: 
        priority = p.get("priority", "P1") 
        planet   = p.get("planet", "") 
        nakshatra= p.get("nakshatra", "") 
        pada     = p.get("pada", 1) 
        rashi    = p.get("rashi", "") 
        weight   = p.get("weight_override", SOURCE_WEIGHTS.get(priority, 1.0)) 
        source_key = p.get("source_key", "")
 
        # HitList 1 — Planet diseases 
        if planet != "—":
            planet_terms = get_planet_terms(planet) 
            for term in planet_terms: 
                if not is_term_allowed(term, gender):
                    continue
                systems = TERM_TO_SYSTEMS.get(term, []) 
                if not systems: 
                    continue 
                hl1_terms.add(term)
                specificity = 1.0 / len(systems) 
                term_weight = (weight * HL_MULTIPLIERS["HL1"]) * specificity
                for system in systems: 
                    system_scores[system] = \
                        system_scores.get(system, 0.0) \
                        + term_weight
                    if priority == "P1" and source_key:
                        p1_system_scores.setdefault(source_key, {})
                        p1_system_scores[source_key][system] = p1_system_scores[source_key].get(system, 0.0) + term_weight
                        p1_system_layers.setdefault(source_key, {})
                        p1_system_layers[source_key].setdefault(system, set()).add("HL1")
                    hitlist_tracker.setdefault( 
                        system, set()).add("HL1") 
                    term_trail.setdefault( 
                        system, []).append({ 
                        "term":     term, 
                        "source":   f"HL1 Planet {planet}", 
                        "priority": priority, 
                        "weight":   round(term_weight, 3) 
                    }) 

        # HitList 2 — Nakshatra diseases 
        if nakshatra != "—":
            nk_terms = get_nakshatra_terms(nakshatra, pada) 
            for term in nk_terms: 
                if not is_term_allowed(term, gender):
                    continue
                systems = TERM_TO_SYSTEMS.get(term, []) 
                if not systems: 
                    continue 
                hl2_terms.add(term)
                specificity = 1.0 / len(systems) 
                term_weight = (weight * HL_MULTIPLIERS["HL2"]) * specificity
                for system in systems: 
                    system_scores[system] = \
                        system_scores.get(system, 0.0) \
                        + term_weight
                    if priority == "P1" and source_key:
                        p1_system_scores.setdefault(source_key, {})
                        p1_system_scores[source_key][system] = p1_system_scores[source_key].get(system, 0.0) + term_weight
                        p1_system_layers.setdefault(source_key, {})
                        p1_system_layers[source_key].setdefault(system, set()).add("HL2")
                    hitlist_tracker.setdefault( 
                        system, set()).add("HL2") 
                    term_trail.setdefault( 
                        system, []).append({ 
                        "term":     term, 
                        "source":   f"HL2 Nakshatra {nakshatra} P{pada}", 
                        "priority": priority, 
                        "weight":   round(term_weight, 3) 
                    }) 

        # HitList 3 — Rashi organs (For P1, P5, P6, P7 or P2 with show_rashi flag)
        if not p.get("suppress_rashi") and (priority in ["P1", "P5", "P6", "P7"] or p.get("show_rashi")):
            rashi_terms = get_rashi_terms(rashi) 
            for term in rashi_terms: 
                if not is_term_allowed(term, gender):
                    continue
                systems = TERM_TO_SYSTEMS.get(term, []) 
                if not systems: 
                    continue 
                hl3_terms.add(term)
                specificity = 1.0 / len(systems) 
                term_weight = (weight * HL_MULTIPLIERS["HL3"]) * specificity
                for system in systems: 
                    system_scores[system] = \
                        system_scores.get(system, 0.0) \
                        + term_weight
                    if priority == "P1" and source_key:
                        p1_system_scores.setdefault(source_key, {})
                        p1_system_scores[source_key][system] = p1_system_scores[source_key].get(system, 0.0) + term_weight
                        p1_system_layers.setdefault(source_key, {})
                        p1_system_layers[source_key].setdefault(system, set()).add("HL3")
                    hitlist_tracker.setdefault( 
                        system, set()).add("HL3") 
                    term_trail.setdefault( 
                        system, []).append({ 
                        "term":     term, 
                        "source":   f"HL3 Rashi {rashi}", 
                        "priority": priority, 
                        "weight":   round(term_weight, 3) 
                    }) 
 
    # Apply a soft coherence penalty only to P1 rows.
    for source_key, system_map in p1_system_scores.items():
        for system, raw_p1_score in system_map.items():
            active_layers = len((p1_system_layers.get(source_key, {}) or {}).get(system, set()))
            coherence_multiplier = P1_COHERENCE_MULTIPLIERS.get(active_layers, 0.5)
            adjusted_p1_score = raw_p1_score * coherence_multiplier
            penalty = raw_p1_score - adjusted_p1_score
            if penalty > 0:
                system_scores[system] = max(0.0, system_scores.get(system, 0.0) - penalty)
                term_trail.setdefault(system, []).append({
                    "term": "P1 coherence penalty",
                    "source": f"P1 Internal Match ({active_layers}/3 layers)",
                    "priority": "P1",
                    "weight": round(-penalty, 3)
                })

    # Apply convergence multiplier 
    final_scores = {} 
    for system, raw_score in system_scores.items(): 
        hitlists_count = len( 
            hitlist_tracker.get(system, set())) 
        multiplier = CONVERGENCE.get( 
            hitlists_count, 1.0) 
        final_scores[system] = round( 
            raw_score * multiplier, 2) 
 
    def _with_hl2_fallback(trail_rows):
        """
        UI shows "Most Probable Diseases" from HL2/Nakshatra trail entries only.
        For clusters formed from HL1 + HL3 without any real HL2 match, inject
        display-only fallback rows so the diseases area is not blank.
        """
        has_real_hl2 = any("HL2" in (row.get("source", "")) for row in trail_rows)
        if has_real_hl2:
            return trail_rows

        fallback_rows = []
        seen_terms = set()
        for row in trail_rows:
            source = row.get("source", "")
            term = row.get("term", "")
            if not term or term in seen_terms:
                continue
            if "HL1" not in source and "Planet" not in source:
                continue
            seen_terms.add(term)
            fallback_rows.append({
                "term": term,
                "source": f"HL2 Fallback from {source}",
                "priority": row.get("priority", ""),
                "weight": 0.0,
            })
            if len(fallback_rows) >= 5:
                break

        return trail_rows + fallback_rows if fallback_rows else trail_rows

    # Merge related systems into display groups 
    group_scores = {} 
    group_trails = {} 
    group_hitlists = {} 
 
    for group_id, group_data in MERGE_GROUPS.items(): 
        systems_in_group = group_data["systems"] 
        label = group_data["label"] 
 
        total = sum( 
            final_scores.get(s, 0.0) 
            for s in systems_in_group 
        ) 
        if total == 0: 
            continue 
 
        combined_hitlists = set() 
        for s in systems_in_group: 
            combined_hitlists.update( 
                hitlist_tracker.get(s, set())) 
 
        combined_trails = [] 
        for s in systems_in_group: 
            combined_trails.extend( 
                term_trail.get(s, [])) 
        combined_trails = _with_hl2_fallback(combined_trails)
 
        group_scores[group_id] = { 
            "label":    label, 
            "score":    round(total, 2), 
            "hitlists": sorted(list(combined_hitlists)), 
            "convergence": len(combined_hitlists), 
            "trail":    combined_trails 
        } 
 
    # Sort by score descending 
    ranked = sorted( 
        group_scores.values(), 
        key=lambda x: x["score"], 
        reverse=True 
    ) 
 
    # Return top 4 
    return { 
        "top4":          ranked[:4], 
        "all_groups":    ranked, 
        "system_scores": final_scores, 
        "hl1": sorted(list(hl1_terms)),
        "hl2": sorted(list(hl2_terms)),
        "hl3": sorted(list(hl3_terms)),
    } 
 
 
def run_hitlist_analysis(chart_data, rule1_result, gender=None):
    """
    Main entry point called from server.py.
    Builds priority_planets list from chart data 
    and rule1_result, then runs hitlist engine.
    """
    from knowledge_base import RASHI_ORGANS 
    gender = normalize_gender(gender or chart_data.get("gender"))

    planets_by_name = { 
        p["name"]: p 
        for p in chart_data.get("planets", []) 
    } 

    priority_planets = [] 

    # ── P1: All planets in 6th house ────────── 
    rk1_list = rule1_result.get("rogkaraka_1", []) 
    has_p1 = False
    for rk in rk1_list: 
        name = rk.get("name", "") 
        p    = planets_by_name.get(name, {}) 
        nk   = p.get("nakshatra", {}) 
        priority_planets.append({ 
            "priority":  "P1", 
            "planet":    name, 
            "nakshatra": nk.get("name", ""), 
            "pada":      nk.get("pada", 1), 
            "rashi":     p.get("sign", {}).get("name", ""), 
            "source_key": f"P1:{name}:{nk.get('name', '')}:{nk.get('pada', 1)}:{p.get('sign', {}).get('name', '')}:{len(priority_planets)}",
        }) 
        has_p1 = True

    if not has_p1:
        for rk in rule1_result.get("eleventh_house_occupants", []) or []:
            name = rk.get("name", "")
            p    = planets_by_name.get(name, {})
            nk   = p.get("nakshatra", {})
            priority_planets.append({
                "priority":  "P1",
                "planet":    name,
                "nakshatra": nk.get("name", ""),
                "pada":      nk.get("pada", 1),
                "rashi":     p.get("sign", {}).get("name", ""),
                "source_key": f"P1_11:{name}:{nk.get('name', '')}:{nk.get('pada', 1)}:{p.get('sign', {}).get('name', '')}:{len(priority_planets)}",
                "label_override": "11th House Occupant as P1",
                "fallback_reason": "6th house occupant absent; 11th house occupant becomes first priority.",
                "suppress_rashi": True,
            })
            has_p1 = True


    # ── P2: Dispositor of 6th lord ──────────── 
    # If 6th-house occupant and 11th-house fallback are absent, keep old P2 fallback.
    # then only dispositor rashi should show.
    rk3 = rule1_result.get("rogkaraka_3", {}) or {} 
    rk3_name = rk3.get("name", "") 
    if rk3_name: 
        p  = planets_by_name.get(rk3_name, {}) 
        nk = p.get("nakshatra", {}) 
        priority_planets.append({ 
            "priority":  "P2", 
            "planet":    rk3_name, 
            "nakshatra": nk.get("name", ""), 
            "pada":      nk.get("pada", 1), 
            "rashi":     p.get("sign", {}).get("name", ""), 
            "show_rashi": False,
        })

    conditional_moon = rule1_result.get("conditional_moon") or {}
    if conditional_moon.get("include"):
        moon_data = conditional_moon.get("planet", {}) or {}
        moon_role = conditional_moon.get("connected_role", "")
        moon_priority = {
            "occupant": "PM1",
            "dispositor": "PM2",
            "drishti": "PM3",
            "6th_house_lord": "PM4",
        }.get(moon_role, "PM4")
        priority_planets.append({
            "priority": moon_priority,
            "planet": "Moon",
            "nakshatra": moon_data.get("nakshatra", ""),
            "pada": moon_data.get("nakshatra_pada", 1),
            "rashi": moon_data.get("sign", ""),
            "show_rashi": True,
        })

    # ── P3: Drishti planets on 6th house ────── 
    drishti = rule1_result.get("drishti_on_6th", []) 
    for d in drishti: 
        name = d.get("planet", "") 
        p    = planets_by_name.get(name, {}) 
        nk   = p.get("nakshatra", {}) 
        priority_planets.append({ 
            "priority":  "P3", 
            "planet":    name, 
            "nakshatra": nk.get("name", ""), 
            "pada":      nk.get("pada", 1), 
            "rashi":     p.get("sign", {}).get("name", ""), 
        }) 

    # ── P4: 6th house lord ──────────────────── 
    sixth_lord = rule1_result.get( 
        "sixth_house_lord", "") 
    if sixth_lord: 
        p  = planets_by_name.get(sixth_lord, {}) 
        nk = p.get("nakshatra", {}) 
        priority_planets.append({ 
            "priority":  "P4", 
            "planet":    sixth_lord, 
            "nakshatra": nk.get("name", ""), 
            "pada":      nk.get("pada", 1), 
            "rashi":     p.get("sign", {}).get("name", ""), 
        }) 

    # ── P5: Ascendant lord ──────────────────── 
    asc_sign = chart_data.get( 
        "ascendant", {}).get("name", "") 
    asc_lord = rule1_result.get( 
        "ascendant_lord", "") 
    if asc_lord: 
        p  = planets_by_name.get(asc_lord, {}) 
        nk = p.get("nakshatra", {}) 
        priority_planets.append({ 
            "priority":  "P5", 
            "planet":    asc_lord, 
            "nakshatra": nk.get("name", ""), 
            "pada":      nk.get("pada", 1), 
            "rashi":     p.get("sign", {}).get("name", ""), 
        }) 

    # ── P6: 6th House Rashi (Always present) ──
    # USER LOGIC: 6th house rashi should always come in table and hitlist logic
    sixth_house_sign = rule1_result.get("sixth_house_sign", "")
    if sixth_house_sign:
        priority_planets.append({
            "priority": "P6",
            "planet": "—",
            "nakshatra": "—",
            "pada": 1,
            "rashi": sixth_house_sign
        })

    # ── P7: Ascendant Rashi (Always present) ──
    if asc_sign:
        priority_planets.append({
            "priority": "P7",
            "planet": "—",
            "nakshatra": "—",
            "pada": 1,
            "rashi": asc_sign
        })

    result = build_hitlists(priority_planets, gender=gender) 
    from eleventh_lord_confirmation import (
        confirmation_boost,
        confirmation_systems,
        get_eleventh_lord_confirmation_source,
    )
    confirmation_source = get_eleventh_lord_confirmation_source(
        chart_data, rule1_result, gender
    )
    system_matches = confirmation_systems(confirmation_source)
    confirmations = []
    for group in result.get("all_groups", []):
        group_systems = next(
            (
                data.get("systems", [])
                for data in MERGE_GROUPS.values()
                if data.get("label") == group.get("label")
            ),
            [],
        )
        matched_systems = sorted(set(group_systems) & set(system_matches))
        if not matched_systems:
            continue
        boost = confirmation_boost(group.get("score", 0))
        if boost <= 0:
            continue
        matched_diseases = sorted({
            disease
            for system in matched_systems
            for disease in system_matches.get(system, [])
        })
        group["score"] = round(group.get("score", 0) + boost, 2)
        evidence = {
            "label": "11th Lord Nakshatra Confirmation",
            "eleventh_lord": confirmation_source.get("planet", ""),
            "nakshatra": confirmation_source.get("nakshatra", ""),
            "pada": confirmation_source.get("pada", 1),
            "matched_systems": matched_systems,
            "confirmation_diseases": matched_diseases,
            "boost": round(boost, 2),
        }
        group.setdefault("trail", []).append({
            "term": ", ".join(matched_diseases),
            "source": evidence["label"],
            "priority": "11L_NK_CONFIRM",
            "weight": round(boost, 3),
        })
        group["eleventh_lord_confirmation"] = evidence
        confirmations.append({"system": group.get("label", ""), **evidence})
    result["all_groups"].sort(key=lambda item: item.get("score", 0), reverse=True)
    result["top4"] = result["all_groups"][:4]
    result["eleventh_lord_nakshatra_confirmation"] = {
        "source": confirmation_source,
        "matches": confirmations,
    }
    result["priority_planets"] = priority_planets 
    return result
