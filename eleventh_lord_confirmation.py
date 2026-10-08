from difflib import SequenceMatcher

from knowledge_base import GENDER_EXCLUSIVE_TERMS, NAKSHATRA_DISEASES
from rashi_truth_review import NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS
from term_to_systems import TERM_TO_SYSTEMS


CONFIRMATION_RATE = 0.05
GENERIC_TOKENS = {
    "disease", "diseases", "disorder", "disorders", "problem", "problems",
    "trouble", "troubles",
}


def _normalize(value: str) -> str:
    text = str(value or "").strip().lower()
    for old, new in {
        "&": " and ", "/": " ", "-": " ", "(": " ", ")": " ",
        ",": " ", ".": " ", "'": "",
    }.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def _allowed(term: str, gender: str | None) -> bool:
    if gender not in {"M", "F"}:
        return True
    normalized = _normalize(term)
    return not any(
        keyword in normalized and allowed_gender != gender
        for keyword, allowed_gender in GENDER_EXCLUSIVE_TERMS.items()
    )


def _nakshatra_diseases(name: str, pada) -> list[str]:
    data = NAKSHATRA_DISEASES.get(name, {}) or {}
    pada_text = str(pada)
    for key, diseases in data.items():
        if pada_text in [item.strip() for item in str(key).split(",")]:
            return list(diseases)
    return []


def get_eleventh_lord_confirmation_source(
    chart_data: dict,
    rule1_result: dict,
    gender: str | None = None,
) -> dict:
    lord = rule1_result.get("eleventh_house_lord", "")
    planet = next(
        (item for item in chart_data.get("planets", []) if item.get("name") == lord),
        {},
    )
    nakshatra = planet.get("nakshatra", {}) or {}
    name = nakshatra.get("name", "") if isinstance(nakshatra, dict) else ""
    pada = nakshatra.get("pada", 1) if isinstance(nakshatra, dict) else 1
    diseases = [
        disease
        for disease in _nakshatra_diseases(name, pada)
        if _allowed(disease, gender)
    ]
    return {
        "planet": lord,
        "nakshatra": name,
        "pada": pada,
        "diseases": diseases,
    }


def match_disease(candidate: str, confirmation_disease: str) -> dict | None:
    left = _normalize(candidate)
    right = _normalize(confirmation_disease)
    if not left or not right:
        return None
    if left == right:
        return {"match_type": "exact_disease", "similarity": 1.0}

    ratio = SequenceMatcher(None, left, right).ratio()
    left_tokens = {token for token in left.split() if token not in GENERIC_TOKENS}
    right_tokens = {token for token in right.split() if token not in GENERIC_TOKENS}
    shared_tokens = left_tokens & right_tokens
    if left in right or right in left or ratio >= 0.9:
        return {"match_type": "close_disease", "similarity": round(ratio, 4)}
    if shared_tokens and ratio >= 0.82:
        return {"match_type": "close_disease", "similarity": round(ratio, 4)}

    candidate_organs = set(NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(candidate, []))
    confirmation_organs = set(
        NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(confirmation_disease, [])
    )
    shared_organs = sorted(candidate_organs & confirmation_organs)
    if shared_organs:
        return {
            "match_type": "same_organ_family",
            "similarity": round(ratio, 4),
            "shared_organs": shared_organs,
        }
    return None


def find_disease_confirmation(candidate: str, source: dict) -> dict | None:
    best = None
    priority = {"exact_disease": 3, "close_disease": 2, "same_organ_family": 1}
    for confirmation_disease in source.get("diseases", []):
        match = match_disease(candidate, confirmation_disease)
        if not match:
            continue
        evidence = {
            "label": "11th Lord Nakshatra Confirmation",
            "eleventh_lord": source.get("planet", ""),
            "nakshatra": source.get("nakshatra", ""),
            "pada": source.get("pada", 1),
            "candidate_disease": candidate,
            "confirmation_disease": confirmation_disease,
            **match,
        }
        if not best or (
            priority[evidence["match_type"]],
            evidence.get("similarity", 0),
        ) > (
            priority[best["match_type"]],
            best.get("similarity", 0),
        ):
            best = evidence
    return best


def confirmation_boost(score: float) -> float:
    return round(float(score or 0) * CONFIRMATION_RATE, 4)


def confirmation_systems(source: dict) -> dict[str, list[str]]:
    systems = {}
    for disease in source.get("diseases", []):
        for system in TERM_TO_SYSTEMS.get(disease, []):
            systems.setdefault(system, []).append(disease)
    return systems
