"""Reflection agent — audits the engine output for logical issues."""
import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from agentic.schemas import CriticOutput

CRITIC_SYSTEM = """You are a medical-astrology OUTPUT AUDITOR.
You receive a JSON object with a deterministic scoring result from an
astrology health engine. Your job is NOT to invent scores — only to AUDIT.

Check for:
1. LOW-SUPPORT GROUPS: groups whose 'hitlists' list has only 1 entry
   (single-source evidence) — flag them.
2. COHERENCE PENALTY: any P1 penalty rows in 'trail' where
   'term' contains 'coherence penalty' — note the layer count.
3. FALLBACK ROWS: any trail row with source containing 'HL2 Fallback'
   is display-only (weight 0.0). Do NOT treat as evidence.
4. CLOSE TOP-2: if score difference between top-1 and top-2 is < 10%,
   flag as ambiguous.
5. GENDER FILTER: verify no trail term appears that could violate the
   provided gender (you don't need the KB — just note if suspicious).

Return STRICT JSON ONLY:
{
  "issues": ["..."],
  "confidence": 0.0-1.0,
  "notes": "one paragraph",
  "suggested_profile": "clinical" | "planet_first" | "consensus_first" | null
}
"""


def critique(llm, engine_result: Dict[str, Any], gender: str,
             fallback_confidence: float = 0.6) -> CriticOutput:
    # Trim the result to only what the critic needs (tokens matter)
    payload = {
        "top4": [
            {
                "label": g["label"],
                "score": g["score"],
                "hitlists": g["hitlists"],
                "convergence": g["convergence"],
                "trail_sample": g["trail"][:8],
            }
            for g in engine_result.get("top4", [])
        ],
        "gender": gender,
    }

    try:
        resp = llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(content=json.dumps(payload, default=str)),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        parsed = json.loads(_extract_json(raw))
        return CriticOutput(**parsed)
    except Exception as e:
        # Fail-safe: neutral critique so pipeline continues. The caller passes its own
        # confidence threshold, so a failed critic never triggers a re-score.
        return CriticOutput(
            issues=[f"critic_failed: {e}"],
            confidence=fallback_confidence,
            notes="Critic unavailable; using the confidence threshold so no re-score is triggered.",
        )


def _extract_json(text: str) -> str:
    """Extract the first {...} block from LLM text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    return text[start:end + 1]