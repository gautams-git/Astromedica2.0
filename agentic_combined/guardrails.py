"""Block invented terms in the LLM summary, and add a disclaimer."""
import re
from typing import Iterable, List

from agentic_combined.schemas import GuardrailReport
from knowledge_base import is_term_allowed_for_gender
from rashi_truth_review import ALL_ORGANS, NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS

DISCLAIMER = (
    "\n\n⚠️ This is an astrological analysis for informational purposes only. "
    "It is NOT a medical diagnosis. Consult a qualified healthcare professional "
    "for any health concerns."
)

# Terms are looked up as whole words (so "Ear" does not match inside "heart").
_DISEASES = sorted({d for d in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS if len(d) >= 5}, key=len, reverse=True)
_ORGANS = sorted({str(o) for o in ALL_ORGANS if len(str(o)) >= 4}, key=len, reverse=True)


def _mentioned(text: str, vocabulary: Iterable[str]) -> List[str]:
    lower = text.lower()
    found = []
    for term in vocabulary:
        if re.search(r"(?<![a-z])" + re.escape(term.lower()) + r"(?![a-z])", lower):
            found.append(term)
    return found


def validate_output(
    summary: str,
    allowed_diseases: Iterable[str],
    allowed_organs: Iterable[str],
    gender: str = "",
    strict: bool = True,
) -> GuardrailReport:
    """The summary may only name diseases and organs that were in the data it was given."""
    report = GuardrailReport(valid=True)
    allowed_text = " | ".join(str(t).lower() for t in [*allowed_diseases, *allowed_organs])

    for term in _mentioned(summary, [*_DISEASES, *_ORGANS]):
        if term.lower() not in allowed_text:
            report.hallucinated_terms.append(term)
        if not is_term_allowed_for_gender(term, gender):
            report.gender_violations.append(term)

    report.hallucinated_terms = list(dict.fromkeys(report.hallucinated_terms))
    report.gender_violations = list(dict.fromkeys(report.gender_violations))

    if report.hallucinated_terms:
        report.errors.append(f"{len(report.hallucinated_terms)} term(s) not in the data")
    if report.gender_violations:
        report.errors.append(f"{len(report.gender_violations)} gender-violating term(s)")

    if strict and (report.hallucinated_terms or report.gender_violations):
        report.valid = False
    return report


def add_disclaimer(text: str) -> str:
    if "not a medical diagnosis" in text.lower():
        return text
    return text + DISCLAIMER
