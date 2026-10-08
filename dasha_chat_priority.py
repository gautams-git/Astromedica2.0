from __future__ import annotations

from knowledge_base import get_nakshatra_diseases


def _dedupe(items):
    seen = set()
    out = []
    for item in items:
        clean = str(item or "").strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _norm(value):
    return str(value or "").strip().lower()


def _planet_name_from_source(source: str) -> str:
    source = str(source or "").strip()
    if "[" in source and "]" in source:
        left = source.split("[", 1)[0].strip()
        return left if left and left.lower() != "unknown" else ""
    return ""


def build_dasha_chat_priority(rule1: dict, dasha: dict, complete_analysis: dict, planets: list[dict]) -> dict:
    maha = (dasha or {}).get("mahadasha", {}) or {}
    antar = (dasha or {}).get("antardasha", {}) or {}
    pratya = (dasha or {}).get("pratyantardasha", {}) or {}

    maha_planet = maha.get("planet", "")
    antar_planet = antar.get("planet", "")
    pratya_planet = pratya.get("planet", "")
    maha_antar = _dedupe([maha_planet, antar_planet])
    all_current = _dedupe([maha_planet, antar_planet, pratya_planet])

    p1_planets = _dedupe(
        [item.get("name", "") for item in (rule1 or {}).get("rogkaraka_1", [])]
        + [((rule1 or {}).get("rogkaraka_2") or {}).get("name", "")]
        + [((rule1 or {}).get("eighth_house_lord_obj") or {}).get("name", "")]
    )
    p2_planets = _dedupe([item.get("planet", "") for item in (rule1 or {}).get("drishti_on_6th", [])])

    top_themes = (complete_analysis or {}).get("most_probable", [])[:3]
    candidate_planets = planets or []
    disease_traces = []
    disease_linked_planets = []

    for theme in top_themes:
        condition = theme.get("condition", "")
        symptoms = theme.get("contributing_symptoms", []) or []
        matched_planets = []

        for planet in candidate_planets:
            nk = planet.get("nakshatra", {}) or {}
            nk_name = nk.get("name", "") if isinstance(nk, dict) else str(nk or "")
            nk_pada = nk.get("pada", 1) if isinstance(nk, dict) else planet.get("nakshatra_pada", 1)
            nk_diseases = get_nakshatra_diseases(nk_name, nk_pada) or []
            matched_symptoms = [symptom for symptom in symptoms if any(_norm(symptom) == _norm(disease) for disease in nk_diseases)]
            if not matched_symptoms:
                continue
            matched_planets.append({
                "planet": planet.get("name", ""),
                "nakshatra": nk_name,
                "pada": nk_pada,
                "matched_symptoms": matched_symptoms,
                "nakshatra_diseases": nk_diseases[:8],
            })
            disease_linked_planets.append(planet.get("name", ""))

        if not matched_planets:
            confirmed_by = theme.get("confirmed_by", "")
            backed_by = theme.get("backed_by", "")
            probable_planets = _dedupe([
                _planet_name_from_source(part.strip())
                for part in (confirmed_by + "," + backed_by).split(",")
            ])
            for name in probable_planets:
                if name:
                    disease_linked_planets.append(name)
            matched_planets = [{"planet": name, "source": "theme_summary"} for name in probable_planets if name]

        disease_traces.append({
            "condition": condition,
            "label": theme.get("label", ""),
            "score": theme.get("score", 0),
            "symptoms": symptoms,
            "matched_planets": matched_planets,
        })

    disease_linked_planets = _dedupe(disease_linked_planets)

    p1_maha_antar = [planet for planet in p1_planets if planet in maha_antar]
    p2_maha_antar = [planet for planet in p2_planets if planet in maha_antar]
    disease_maha_antar = [planet for planet in disease_linked_planets if planet in maha_antar]
    p1_pratya = [planet for planet in p1_planets if planet == pratya_planet]
    p2_pratya = [planet for planet in p2_planets if planet == pratya_planet]
    disease_pratya = [planet for planet in disease_linked_planets if planet == pratya_planet]

    risk_level = "very_low"
    chat_mode = "general_lifestyle_first"
    explanation = "No P1, P2, or traced disease-linked planets are active in the current dasha. Start broad and gently."

    if p1_maha_antar and disease_maha_antar:
        risk_level = "extra_high"
        chat_mode = "direct_disease_first"
        explanation = "P1 planets are active in Mahadasha/Antardasha and traced disease-linked planets are also active there."
    elif p1_maha_antar:
        risk_level = "very_high"
        chat_mode = "direct_disease_first"
        explanation = "A P1 planet (6th lord, 8th lord, or 6th-house occupant) is active in Mahadasha/Antardasha."
    elif p2_maha_antar or disease_maha_antar:
        risk_level = "high"
        chat_mode = "focused_symptom_first"
        explanation = "A P2 drishti planet or disease-linked nakshatra planet is active in Mahadasha/Antardasha."
    elif p1_pratya or p2_pratya or disease_pratya:
        risk_level = "medium"
        chat_mode = "mild_symptom_first"
        explanation = "No strong Maha/Antar match was found, but an important trigger is active in Pratyantardasha."

    return {
        "risk_level": risk_level,
        "chat_mode": chat_mode,
        "explanation": explanation,
        "current_dasha": {
            "mahadasha": maha_planet,
            "antardasha": antar_planet,
            "pratyantardasha": pratya_planet,
        },
        "priority_groups": {
            "p1": p1_planets,
            "p2": p2_planets,
            "disease_linked": disease_linked_planets,
        },
        "matches": {
            "p1_maha_antar": p1_maha_antar,
            "p2_maha_antar": p2_maha_antar,
            "disease_maha_antar": disease_maha_antar,
            "p1_pratyantardasha": p1_pratya,
            "p2_pratyantardasha": p2_pratya,
            "disease_pratyantardasha": disease_pratya,
        },
        "top_hitlist_themes": disease_traces,
        "backend_summary": {
            "maha_antar_checked": maha_antar,
            "all_current_checked": all_current,
        },
    }
