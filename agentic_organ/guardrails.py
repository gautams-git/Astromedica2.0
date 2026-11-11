"""Guardrails for organ/disease terms."""
from typing import Dict, Set

from agentic_organ.schemas import GuardrailReport
from knowledge_base import is_term_allowed_for_gender


DISCLAIMER = (
    "\n\n⚠️ This is an astrological analysis for informational purposes only. "
    "It is NOT a medical diagnosis. Consult a qualified healthcare professional "
    "for any health concerns."
)


def _collect_engine_terms(result: Dict) -> Set[str]:
    out = set()
    for o in result.get("all_organs", []):
        out.add(o["organ"])
        for d in o.get("related_diseases", []):
            out.add(d["disease"])
    for d in result.get("all_diseases", []):
        out.add(d["disease"])
    return out


def validate_output(
    summary: str,
    mentioned_terms: list,
    engine_result: Dict,
    gender: str,
    strict: bool = True,
) -> GuardrailReport:
    report = GuardrailReport(valid=True)
    allowed = _collect_engine_terms(engine_result)

    for term in mentioned_terms or []:
        if term not in allowed:
            if len(term.split()) > 2:
                report.hallucinated_diseases.append(term)
            else:
                report.hallucinated_organs.append(term)
        if not is_term_allowed_for_gender(term, gender):
            report.gender_violations.append(term)

    if report.hallucinated_organs:
        report.errors.append(
            f"{len(report.hallucinated_organs)} hallucinated organ(s)"
        )
    if report.hallucinated_diseases:
        report.errors.append(
            f"{len(report.hallucinated_diseases)} hallucinated disease(s)"
        )
    if report.gender_violations:
        report.errors.append(
            f"{len(report.gender_violations)} gender-violating term(s)"
        )

    if strict and (
        report.hallucinated_organs
        or report.hallucinated_diseases
        or report.gender_violations
    ):
        report.valid = False

    return report


def add_disclaimer(text: str) -> str:
    if "not a medical diagnosis" in text.lower():
        return text
    return text + DISCLAIMER