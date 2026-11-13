"""Block invented terms and gender violations in the LLM summary, and add a disclaimer."""
import re
from typing import Iterable, List

from agentic_disease_first.guardrails import DISCLAIMER, add_disclaimer   # same wording everywhere
from knowledge_base import is_term_allowed_for_gender
from term_to_systems import TERM_TO_SYSTEMS

from agentic_rashi_correlation.schemas import GuardrailReport

# The engine's own vocabulary of disease / organ terms. Terms are looked up as whole words
# (so "Ear" does not match inside "heart").
_VOCABULARY = sorted({str(t) for t in TERM_TO_SYSTEMS if len(str(t)) >= 4}, key=len, reverse=True)


def _mentioned(text: str, vocabulary: Iterable[str]) -> List[str]:
    lower = text.lower()
    found = []
    for term in vocabulary:
        if re.search(r"(?<![a-z])" + re.escape(term.lower()) + r"(?![a-z])", lower):
            found.append(term)
    return found


def validate_output(
    summary: str,
    allowed_terms: Iterable[str],
    gender: str,
    strict: bool = True,
    vocabulary: Iterable[str] | None = None,
) -> GuardrailReport:
    """The summary may only name diseases and organs that were in the data it was given.

    allowed_terms: the group labels, organs and diseases that were sent to the summarizer.
    """
    report = GuardrailReport(valid=True)
    allowed_text = " | ".join(str(t).lower() for t in allowed_terms)

    for term in _mentioned(summary, _VOCABULARY if vocabulary is None else vocabulary):
        # "Kidney" is fine when "Kidney trouble" was given (substring either way)
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


__all__ = ["DISCLAIMER", "add_disclaimer", "validate_output"]
