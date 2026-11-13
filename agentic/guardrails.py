"""Block hallucinations, gender violations, and add disclaimers."""
from typing import Any, Dict, Set

from agentic.schemas import GuardrailReport
from hitlist_engine import is_term_allowed
from term_to_systems import TERM_TO_SYSTEMS

DISCLAIMER = (
    "\n\n⚠️ This is an astrological analysis for informational purposes only. "
    "It is NOT a medical diagnosis. Consult a qualified healthcare professional "
    "for any health concerns."
)


def _collect_all_engine_terms(result: Dict) -> Set[str]:
    terms = set()
    for group in result.get("all_groups", []):
        for row in group.get("trail", []):
            t = row.get("term", "")
            if t and "Fallback" not in row.get("source", "") and "Penalty" not in t:
                terms.add(t)
    for hl in ("hl1", "hl2", "hl3"):
        terms.update(result.get(hl, []))
    return terms


def validate_output(
    summary: str,
    mentioned_terms: list,
    engine_result: Dict,
    gender: str,
    strict: bool = True,
) -> GuardrailReport:
    """Ensure LLM summary doesn't invent terms or bypass gender filter."""
    report = GuardrailReport(valid=True)

    allowed_terms = _collect_all_engine_terms(engine_result)
    kb_terms = set(TERM_TO_SYSTEMS.keys())

    for term in mentioned_terms or []:
        if term not in allowed_terms:
            report.hallucinated_terms.append(term)
        if term not in kb_terms:
            report.hallucinated_terms.append(term)
        if not is_term_allowed(term, gender):
            report.gender_violations.append(term)

    if report.hallucinated_terms:
        report.errors.append(
            f"{len(report.hallucinated_terms)} hallucinated term(s) detected"
        )
    if report.gender_violations:
        report.errors.append(
            f"{len(report.gender_violations)} gender-violating term(s)"
        )

    if strict and (report.hallucinated_terms or report.gender_violations):
        report.valid = False

    return report


def add_disclaimer(text: str) -> str:
    if "not a medical diagnosis" in text.lower():
        return text
    return text + DISCLAIMER