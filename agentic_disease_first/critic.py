"""Audit of the engine output: rule-based checks always, plus an optional LLM review."""
import json
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_disease_first.schemas import CriticOutput

CRITIC_SYSTEM = """You are a medical-astrology OUTPUT AUDITOR.
You receive a JSON object with a deterministic scoring result from the
"Disease First Logic" engine (diseases ranked from the nakshatra diseases of the
Rule 1 source planets, backed by planet-organ and rashi-organ support).
Your job is NOT to invent scores — only to AUDIT.

Check for:
1. SINGLE-SOURCE DISEASES: a top disease whose 'source_priorities' has only one entry.
2. CLOSE TOP-2: if the score gap between top-1 and top-2 is < 10%, flag as ambiguous.
3. FALLBACK: 'used_p6_fallback' true means the 11th house occupant replaced a missing
   6th house occupant — note it.
4. ORGAN SUPPORT: a top disease with no 'organs' is weakly supported.
5. GENDER: note any disease or organ that looks wrong for the given gender.
'rule_based_issues' are checks already done in code — do not repeat them, only add new ones.

Return STRICT JSON ONLY:
{
  "issues": ["..."],
  "confidence": 0.0-1.0,
  "notes": "one paragraph",
  "suggested_profile": "clinical" | "planet_heavy" | "rashi_heavy" | null
}
"""


def rule_based_audit(result: Dict[str, Any]) -> Tuple[List[str], float]:
    """Plain-code checks. Deterministic, free, and the fallback when the LLM is unavailable."""
    diseases = result.get("top_diseases") or []
    if not diseases:
        return ["no diseases were scored"], 0.3

    issues: List[str] = []
    confidence = 0.9

    single = [d for d in diseases[:3] if len(d.get("source_priorities") or []) <= 1]
    for d in single:
        priorities = ", ".join(d.get("source_priorities") or ["none"])
        issues.append(f"'{d['disease']}' is backed by a single source ({priorities})")
    confidence -= min(0.15, 0.05 * len(single))

    if len(diseases) >= 2 and float(diseases[0].get("score") or 0) > 0:
        top, second = float(diseases[0]["score"]), float(diseases[1]["score"])
        gap = (top - second) / top
        if gap < 0.10:
            issues.append(f"top 2 diseases are within {gap:.0%} of each other (ambiguous)")
            confidence -= 0.1

    if result.get("used_p6_fallback"):
        issues.append(result.get("fallback_note") or "11th house occupant used instead of the 6th")
        confidence -= 0.1

    if not diseases[0].get("supporting_organs"):
        issues.append(f"'{diseases[0]['disease']}' has no organ support")
        confidence -= 0.1

    return issues, round(max(0.3, min(0.95, confidence)), 2)


def _payload(result: Dict[str, Any], gender: str, rule_issues: List[str]) -> Dict[str, Any]:
    return {
        "top_diseases": [
            {
                "disease": d["disease"],
                "score": d["score"],
                "source_priorities": d.get("source_priorities"),
                "source_labels": d.get("source_labels"),
                "organs": [o["organ"] for o in (d.get("supporting_organs") or [])[:3]],
                "planets": [p["planet"] for p in (d.get("planet_support") or [])],
            }
            for d in (result.get("top_diseases") or [])[:5]
        ],
        "used_p6_fallback": result.get("used_p6_fallback"),
        "sources_used": result.get("sources_used"),
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
