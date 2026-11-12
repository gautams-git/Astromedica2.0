"""Audit of the Combined Algo output: rule-based checks always, plus an optional LLM review."""
import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_combined.schemas import CriticOutput

TOTAL_ALGORITHMS = 5
MEDIAN_NEEDS = 3         # the median of 5 scores only becomes positive when 3 algorithms vote

CRITIC_SYSTEM = """You are a medical-astrology OUTPUT AUDITOR.
You receive the result of the "Combined Algo": the median consensus of five algorithms.
Every organ has a score from each algorithm (0 = that algorithm did not mention it) and the
consensus score is the MEDIAN of the five. The top 3 organs are returned, each with up to 3
diseases and the algorithms that named them.
Your job is NOT to invent scores - only to AUDIT.

Check for:
1. WEAK AGREEMENT: a top organ named by fewer than 3 of the 5 algorithms.
2. CLOSE TOP-2: consensus gap between organ 1 and organ 2 below 10%.
3. MISSING ALGORITHMS: algorithms listed in 'algorithms_without_data' did not vote.
4. DISEASES: a top organ with no diseases, or whose diseases all come from one algorithm.
5. GENDER: note any organ or disease that looks wrong for a typical patient context.
'rule_based_issues' are checks already done in code - do not repeat them, only add new ones.

Return STRICT JSON ONLY:
{
  "issues": ["..."],
  "confidence": 0.0-1.0,
  "notes": "one paragraph"
}
"""


def rule_based_audit(result: Dict[str, Any], algorithms_with_data: Optional[Dict[str, bool]] = None) -> Tuple[List[str], float]:
    """Plain-code checks. Deterministic, free, and the fallback when the LLM is unavailable."""
    organs = result.get("top_organs") or []
    if not organs:
        return ["no organ received enough support from the algorithms"], 0.3

    issues: List[str] = []
    confidence = 0.9

    top = organs[0]
    support = int(top.get("algorithm_support_count") or 0)
    if support < MEDIAN_NEEDS:
        issues.append(f"'{top['organ']}' is named by only {support} of {TOTAL_ALGORITHMS} algorithms (the median needs {MEDIAN_NEEDS})")
        confidence -= 0.25 if support <= 1 else 0.15
    elif support == MEDIAN_NEEDS:
        confidence -= 0.03

    if len(organs) >= 2 and float(top.get("consensus_score") or 0) > 0:
        gap = (float(top["consensus_score"]) - float(organs[1].get("consensus_score") or 0)) / float(top["consensus_score"])
        if gap < 0.10:
            issues.append(f"top 2 organs are within {gap:.0%} of each other (ambiguous)")
            confidence -= 0.1

    if len(organs) < 3:
        issues.append(f"only {len(organs)} organ(s) reached the ranking (3 expected)")
        confidence -= 0.05

    missing = [name for name, ok in (algorithms_with_data or {}).items() if not ok]
    for name in missing:
        issues.append(f"{name} gave no ranking, so it did not vote")
    confidence -= min(0.2, 0.1 * len(missing))

    if not top.get("top_diseases"):
        issues.append(f"'{top['organ']}' has no diseases attached")
        confidence -= 0.1
    else:
        sources = {s for d in top["top_diseases"] for s in (d.get("sources") or [])}
        if len(sources) <= 1:
            issues.append(f"all diseases of '{top['organ']}' come from a single algorithm")
            confidence -= 0.05

    return issues, round(max(0.3, min(0.95, confidence)), 2)


def _payload(result: Dict[str, Any], algorithms_with_data: Dict[str, bool], rule_issues: List[str]) -> Dict[str, Any]:
    return {
        "top_organs": [
            {
                "organ": o["organ"],
                "consensus_score": o.get("consensus_score"),
                "algorithm_support_count": o.get("algorithm_support_count"),
                "algorithm_scores": o.get("algorithm_scores"),
                "diseases": [{"disease": d["disease"], "sources": d.get("sources")} for d in (o.get("top_diseases") or [])],
            }
            for o in (result.get("top_organs") or [])
        ],
        "algorithms_without_data": [n for n, ok in algorithms_with_data.items() if not ok],
        "rule_based_issues": rule_issues,
    }


def critique(llm, engine_result: Dict[str, Any], algorithms_with_data: Optional[Dict[str, bool]] = None,
             use_llm: bool = True) -> CriticOutput:
    algorithms_with_data = algorithms_with_data or {}
    rule_issues, baseline = rule_based_audit(engine_result, algorithms_with_data)

    if not use_llm:
        return CriticOutput(
            issues=rule_issues,
            confidence=baseline,
            notes="Rule-based audit (no live LLM configured).",
        )

    try:
        resp = llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(content=json.dumps(_payload(engine_result, algorithms_with_data, rule_issues), default=str)),
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
    )


def _extract_json(text: str) -> str:
    """Extract the first {...} block from LLM text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    return text[start:end + 1]
