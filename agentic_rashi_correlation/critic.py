"""Audit of the Rashi engine output: rule-based checks always, plus an optional LLM review.

The critic only assesses and reports. It never produces new correlations, evidence or scores.
"""
import json
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rashi_correlation.schemas import CriticOutput

TOP_N = 3   # the engine's own top3

CRITIC_SYSTEM = """You are a medical-astrology OUTPUT AUDITOR.
You receive a JSON object with a deterministic result from the "Rashi Correlation" engine:
Rule 1 source evidence (planet, nakshatra and source-rashi terms, ranked by priority P1..P5 and
PM1..PM4) is matched against the organs of two anchors - the 6th house rashi (P6) and the
ascendant rashi (P7) - and the matches are merged into organ-system groups.
Your job is NOT to invent scores, evidence or correlations - only to AUDIT what is given.

For each group in 'top' you get: label, score, anchors (P6/P7 matched), priorities (the
evidence priorities that supported it), organs (matched anchor organs) and diseases
(matched evidence terms).

Check for:
1. SINGLE-PRIORITY GROUPS: a top group whose 'priorities' has only one entry.
2. CLOSE TOP-2: if the score gap between top-1 and top-2 is < 10%, flag as ambiguous.
3. ANCHOR COVERAGE: a top group matched against only one anchor (P6 or P7) is weaker than one
   matched against both; say so when it matters for the ranking.
4. MISSING EVIDENCE: a top group with no 'organs' or no 'diseases' is weakly supported.
5. CONSISTENCY: the group label, its organs and its diseases should belong together; flag a
   group whose organs/diseases do not fit its label.
6. FALLBACK: a non-empty 'fallback_note' means the 11th house occupant replaced a missing
   6th house occupant - note it.
7. GENDER: note any disease or organ that looks wrong for the given gender.
'rule_based_issues' are checks already done in code - do not repeat them, only add new ones.

HARD RULES:
- Use ONLY what is in the JSON. Do NOT invent planetary evidence, diseases, organs or
  organ associations, and do NOT add astrological interpretations that the data does not support.
- Do NOT change or re-rank the groups.

Return STRICT JSON ONLY:
{
  "issues": ["..."],
  "confidence": 0.0-1.0,
  "notes": "one paragraph",
  "suggested_profile": "clinical" | "nakshatra_heavy" | "anchor_heavy" | null
}
"""


def rule_based_audit(result: Dict[str, Any]) -> Tuple[List[str], float]:
    """Plain-code checks. Deterministic, free, and the fallback when the LLM is unavailable."""
    groups = result.get("top3") or []

    if not groups:
        return ["no groups were scored"], 0.3

    issues, confidence = [], 0.9

    single = [
        g for g in groups
        if len(g.get("supporting_priorities") or []) <= 1
    ]

    for g in single:
        issues.append(
            f"'{g['label']}' is backed by a single priority"
        )

    confidence -= min(0.15, 0.05 * len(single))

    if len(groups) >= 2 and float(groups[0]["score"]) > 0:
        gap = (
            groups[0]["score"] - groups[1]["score"]
        ) / groups[0]["score"]

        if gap < 0.10:
            issues.append(
                f"top 2 groups are within {gap:.0%} of each other (ambiguous)"
            )
            confidence -= 0.1

    if result.get("fallback_note"):
        issues.append(result["fallback_note"])
        confidence -= 0.1

    if not groups[0].get("matched_organs"):
        issues.append(
            f"'{groups[0]['label']}' has no matched organs"
        )
        confidence -= 0.1

    return issues, round(max(0.3, min(0.95, confidence)), 2)


def top_groups_payload(result: Dict[str, Any], n: int = TOP_N) -> List[Dict[str, Any]]:
    """The evidence handed to the critic and the summarizer. Field names are fixed."""
    return [
        {
            "label": g["label"],
            "score": g["score"],
            "anchors": g.get("matched_anchors") or [],
            "priorities": g.get("supporting_priorities") or [],
            "organs": (g.get("matched_organs") or [])[:5],
            "diseases": (g.get("matched_diseases") or [])[:5],
        }
        for g in (result.get("top3") or [])[:n]
    ]


def anchor_signs(result: Dict[str, Any]) -> Dict[str, str]:
    return {key: (a or {}).get("sign", "") for key, a in (result.get("anchors") or {}).items()}


def _payload(result: Dict[str, Any], gender: str, rule_issues: List[str]) -> Dict[str, Any]:
    return {
        "top": top_groups_payload(result),
        "anchor_signs": anchor_signs(result),
        "fallback_note": result.get("fallback_note") or "",
        "gender": gender,
        "rule_based_issues": rule_issues,
    }


def critique(llm, engine_result: Dict[str, Any], gender: str, use_llm: bool = True) -> CriticOutput:
    rule_issues, baseline = rule_based_audit(engine_result)

    if not use_llm:
        return CriticOutput(
            issues=rule_issues,
            confidence=baseline,
            notes="Rule-based audit (no live LLM configured).",
        )

    try:
        resp = llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(content=json.dumps(_payload(engine_result, gender, rule_issues), default=str)),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(raw, list):   # some chat models return content parts
            raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
        out = CriticOutput(**json.loads(_extract_json(raw)))
    except Exception as e:
        # The rule-based audit is the fail-safe, so a broken LLM never blocks the chart.
        return CriticOutput(
            issues=rule_issues + [f"llm_critic_failed: {e}"],
            confidence=baseline,
            notes="LLM critic unavailable; rule-based audit used.",
        )

    return CriticOutput(
        issues=list(dict.fromkeys(rule_issues + out.issues)),
        confidence=out.confidence,
        notes=out.notes,
        suggested_profile=out.suggested_profile,
    )


def _extract_json(text: str) -> str:
    """Extract the first {...} block from LLM text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    return text[start:end + 1]
