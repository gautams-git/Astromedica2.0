"""Independent arithmetic check for Rashi Correlation.

This module rebuilds the published contribution formula from the chart and the
weight tables that are active at call time. It does not call
run_rashi_correlation_analysis. A mismatch is a finding, not a reason to edit
the engine.

The 11th-lord confirmation matcher is the existing confirmation module. The
boost arithmetic (score * 0.05, rounded) is checked here. If the trail does
not expose enough fields, the result is marked limited instead of passed.
"""

from __future__ import annotations

from typing import Any

import rashi_correlation_engine as rce
from eleventh_lord_confirmation import (
    confirmation_boost,
    find_disease_confirmation,
    get_eleventh_lord_confirmation_source,
)
from knowledge_base import (
    NAKSHATRA_DISEASES,
    PLANET_DISEASES,
    RASHI_ORGANS,
    is_term_allowed_for_gender,
    normalize_gender,
)
from term_to_systems import MERGE_GROUPS, TERM_TO_SYSTEMS

SKIP_PREFIXES = ("Further everything", "Further all")
GENERIC_SHARED_SYSTEMS = {"dermatological", "sensory", "systemic"}
SUPPORT_ONLY_SYSTEMS = {"blood", "lymphatic"}


def _norm(value: Any) -> str:
    return (value or "").strip().lower()


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        key = _norm(item)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _group_lookup() -> dict[str, dict[str, str]]:
    lookup = {}
    for group_id, group_data in MERGE_GROUPS.items():
        for system in group_data.get("systems", []):
            lookup[system] = {"id": group_id, "label": group_data.get("label", system)}
    return lookup


def _planet_terms(planet_name: str) -> list[str]:
    raw = PLANET_DISEASES.get(planet_name, "")
    terms = []
    for term in raw.split(","):
        clean = term.strip()
        if not clean or any(clean.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        terms.append(clean)
    return terms


def _nakshatra_terms(nakshatra_name: str, pada: Any) -> list[str]:
    data = NAKSHATRA_DISEASES.get(nakshatra_name, {})
    pada_str = str(pada)
    for key, diseases in data.items():
        if pada_str in [part.strip() for part in key.split(",")]:
            return list(diseases)
    return []


def _rashi_terms(rashi_name: str) -> list[str]:
    raw = RASHI_ORGANS.get(rashi_name, "")
    return [term.strip() for term in raw.split(",") if term.strip()]


def _systems(terms: list[str]) -> list[str]:
    systems = set()
    for term in terms:
        for system in TERM_TO_SYSTEMS.get(term, []):
            systems.add(system)
    return sorted(systems)


def _evidence_sources(chart: dict, rule1: dict, gender: str | None) -> list[dict]:
    planets = {planet.get("name"): planet for planet in chart.get("planets", []) if isinstance(planet, dict)}
    sources = []

    def add(priority: str, label: str, planet_name: str, include_rashi: bool) -> None:
        if not planet_name:
            return
        planet = planets.get(planet_name) or {}
        if not isinstance(planet, dict):
            return
        nakshatra = planet.get("nakshatra") or {}
        sign = planet.get("sign") or {}
        if not isinstance(nakshatra, dict):
            nakshatra = {}
        if not isinstance(sign, dict):
            sign = {}
        nak_name = nakshatra.get("name", "")
        pada = nakshatra.get("pada", 1)
        rashi = sign.get("name", "")
        allowed = lambda term: is_term_allowed_for_gender(term, gender)
        sources.append({
            "priority": priority,
            "label": label,
            "planet": planet_name,
            "planet_terms": _dedupe([term for term in _planet_terms(planet_name) if allowed(term)]),
            "nakshatra_terms": _dedupe([term for term in _nakshatra_terms(nak_name, pada) if allowed(term)]),
            "rashi_terms": _dedupe([term for term in _rashi_terms(rashi) if allowed(term)]) if include_rashi else [],
        })

    for item in rule1.get("rogkaraka_1") or []:
        if isinstance(item, dict):
            add("P1", "Occupant of 6th House", item.get("name", ""), False)
    if not rule1.get("rogkaraka_1"):
        for item in rule1.get("eleventh_house_occupants") or []:
            if isinstance(item, dict):
                add("P1", "11th House Occupant as P1", item.get("name", ""), False)
    rk3 = rule1.get("rogkaraka_3") or {}
    if isinstance(rk3, dict):
        add("P2", "Dispositor of 6th Lord", rk3.get("name", ""), True)
    for item in rule1.get("drishti_on_6th") or []:
        if isinstance(item, dict):
            add("P3", "Drishti on 6th House", item.get("planet", ""), True)
    add("P4", "6th House Lord", rule1.get("sixth_house_lord", ""), True)
    add("P5", "Ascendant Lord", rule1.get("ascendant_lord", ""), True)
    moon = rule1.get("conditional_moon") or {}
    if isinstance(moon, dict) and moon.get("include"):
        role = moon.get("connected_role", "")
        priority = {"occupant": "PM1", "dispositor": "PM2", "drishti": "PM3", "6th_house_lord": "PM4"}.get(role, "PM4")
        add(priority, f"Conditional Moon via {str(role).replace('_', ' ')}", "Moon", True)
    return sources


def _score_groups(chart: dict, rule1: dict, gender: str | None) -> tuple[list[dict], list[dict]]:
    gender = normalize_gender(gender or chart.get("gender"))
    lookup = _group_lookup()
    anchors = {}
    for key, sign_name, label in (
        ("P6", rule1.get("sixth_house_sign", ""), "6th House Rashi"),
        ("P7", (chart.get("ascendant") or {}).get("name", "") if isinstance(chart.get("ascendant"), dict) else "", "Ascendant Rashi"),
    ):
        terms = _dedupe([term for term in _rashi_terms(sign_name) if is_term_allowed_for_gender(term, gender)])
        anchors[key] = {"terms": terms, "systems": set(_systems(terms)), "sign": sign_name, "label": label}

    groups: dict[str, dict] = {}
    contributions = []
    for source in _evidence_sources(chart, rule1, gender):
        priority_weight = float(rce.PRIORITY_WEIGHTS.get(source["priority"], 1.0))
        layers = (
            ("planet", source["planet_terms"]),
            ("nakshatra", source["nakshatra_terms"]),
            ("source_rashi", source["rashi_terms"]),
        )
        for anchor_key, anchor in anchors.items():
            if not anchor["terms"]:
                continue
            anchor_weight = float(rce.ANCHOR_WEIGHTS.get(anchor_key, 1.0))
            for layer_name, terms in layers:
                layer_weight = float(rce.LAYER_WEIGHTS.get(layer_name, 1.0))
                for evidence_term in terms:
                    evidence_systems = set(TERM_TO_SYSTEMS.get(evidence_term, []))
                    if not evidence_systems:
                        continue
                    best = None
                    for anchor_term in anchor["terms"]:
                        anchor_systems = set(TERM_TO_SYSTEMS.get(anchor_term, []))
                        if not anchor_systems:
                            continue
                        shared = evidence_systems & anchor_systems
                        exact = _norm(evidence_term) == _norm(anchor_term)
                        if not shared and not exact:
                            continue
                        generic_only = bool(shared) and all(system in GENERIC_SHARED_SYSTEMS for system in shared)
                        support_only = bool(shared) and all(system in SUPPORT_ONLY_SYSTEMS for system in shared)
                        if not exact and generic_only:
                            continue
                        if support_only and layer_name != "nakshatra" and not exact:
                            continue
                        strength = 1.35 if exact else 1.0
                        if shared:
                            strength += min(0.25, 0.08 * len(shared))
                        for system in sorted(shared or evidence_systems or anchor_systems):
                            meta = lookup.get(system)
                            if not meta:
                                continue
                            if not exact and system in GENERIC_SHARED_SYSTEMS:
                                continue
                            if support_only and system in SUPPORT_ONLY_SYSTEMS and layer_name != "nakshatra" and not exact:
                                continue
                            specificity = 1.0 / max(len(shared) if shared else 1, 1)
                            raw = priority_weight * layer_weight * anchor_weight * strength * (1.0 + specificity * 0.2)
                            candidate = {
                                "group_id": meta["id"],
                                "label": meta["label"],
                                "anchor_term": anchor_term,
                                "system": system,
                                "exact": exact,
                                "weight": round(raw, 3),
                                "priority_tuple": (
                                    1 if layer_name == "nakshatra" else 0,
                                    1 if anchor_key == "P6" else 0,
                                    1 if exact else 0,
                                    -len(shared) if shared else -99,
                                    round(raw, 3),
                                ),
                            }
                            if best is None or candidate["priority_tuple"] > best["priority_tuple"]:
                                best = candidate
                    if not best:
                        continue
                    group = groups.setdefault(best["group_id"], {
                        "id": best["group_id"],
                        "label": best["label"],
                        "score": 0.0,
                        "anchors": set(),
                        "priorities": set(),
                        "organs": set(),
                        "diseases": set(),
                    })
                    group["score"] += best["weight"]
                    group["anchors"].add(anchor_key)
                    group["priorities"].add(source["priority"])
                    group["organs"].add(best["anchor_term"])
                    group["diseases"].add(evidence_term)
                    contributions.append({
                        "group": best["label"],
                        "priority": source["priority"],
                        "layer": layer_name,
                        "anchor": anchor_key,
                        "weight": best["weight"],
                        "kind": "match",
                    })

    source = get_eleventh_lord_confirmation_source(chart, rule1, gender)
    for group in groups.values():
        match = None
        for disease in group["diseases"]:
            match = find_disease_confirmation(disease, source)
            if match:
                break
        if not match:
            continue
        boost = confirmation_boost(group["score"])
        if boost <= 0:
            continue
        group["score"] += boost
        contributions.append({
            "group": group["label"],
            "priority": "11L_NK_CONFIRM",
            "layer": "nakshatra_confirmation",
            "anchor": "",
            "weight": round(boost, 3),
            "kind": "confirmation",
        })

    ranked = []
    for group in groups.values():
        ranked.append({
            "label": group["label"],
            "score": round(group["score"], 2),
            "organs": sorted(group["organs"]),
            "diseases": sorted(group["diseases"]),
            "priorities": sorted(group["priorities"]),
        })
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked, contributions


def _trail_sum(engine: dict) -> dict[str, float]:
    totals = {}
    for group in engine.get("all_groups") or []:
        label = group.get("label")
        total = 0.0
        countable = False
        for step in group.get("trail") or []:
            if "weight" not in step:
                continue
            countable = True
            total += float(step["weight"])
        if countable:
            totals[label] = round(total, 2)
    return totals


def verify_rashi(chart: dict, rule1: dict, gender: str | None, engine: dict | None) -> dict[str, Any]:
    """Compare an independent reconstruction with the engine report."""
    if not isinstance(engine, dict) or "all_groups" not in engine:
        return {
            "status": "limited",
            "reason": "The engine output does not expose all_groups, so the score was not reconstructed.",
            "comparisons": [],
        }
    try:
        ranked, contributions = _score_groups(chart, rule1 or {}, gender)
    except Exception as exc:
        return {
            "status": "limited",
            "reason": f"Independent reconstruction could not read this input ({type(exc).__name__}: {exc}).",
            "comparisons": [],
        }

    reported = [
        {"label": item.get("label"), "score": item.get("score")}
        for item in engine.get("all_groups") or []
    ]
    expected = [{"label": item["label"], "score": item["score"]} for item in ranked]
    mismatches = []
    for index, (left, right) in enumerate(zip(reported, expected)):
        if left["label"] != right["label"] or abs(float(left["score"]) - float(right["score"])) > 0.02:
            mismatches.append({"rank": index + 1, "engine": left, "oracle": right})
    if len(reported) != len(expected):
        mismatches.append({"rank": "length", "engine": len(reported), "oracle": len(expected)})

    trail = _trail_sum(engine)
    trail_issues = []
    for item in reported:
        if item["label"] in trail and abs(float(item["score"]) - trail[item["label"]]) > 0.05:
            trail_issues.append({
                "label": item["label"],
                "reported_score": item["score"],
                "trail_sum": trail[item["label"]],
            })

    status = "passed" if not mismatches and not trail_issues else "mismatch"
    return {
        "status": status,
        "reason": "Independent contribution sum matches the engine." if status == "passed" else "Arithmetic or ordering differs from the engine report.",
        "expected": expected,
        "reported": reported,
        "mismatches": mismatches,
        "trail_issues": trail_issues,
        "contribution_count": len(contributions),
        "contributions_top": contributions[:12],
    }
