"""Reflection agent — audits organ-truth results (no weight changes)."""
import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_organ.schemas import CriticOutput

CRITIC_SYSTEM = """You are a medical-astrology OUTPUT AUDITOR for an
ORGAN ↔ DISEASE correlation engine.

You receive top organs and top diseases with scores, anchor hits,
and source priorities. You do NOT invent scores or terms — you AUDIT.

Check for:
1. SINGLE-SOURCE ORGANS: organs supported by only one source priority.
2. ANCHOR MISMATCH: top organs that have no anchor_hits.
3. DISEASE-ORGAN MISMATCH: a disease in top_diseases whose
   matched_organs do not appear in top_organs (weak grounding).
4. CLOSE TOP-2: score difference < 10% between first and second organ.
5. EMPTY SIGNALS: top_organs or top_diseases is empty while anchors exist.

Return STRICT JSON ONLY:
{
  "issues": ["..."],
  "confidence": 0.0-1.0,
  "notes": "one paragraph",
  "request_rerun": false
}

Set "request_rerun": true ONLY if the result looks structurally broken
(e.g., empty top_organs despite valid anchors) and a re-run with identical
inputs might fix a transient issue. Otherwise false.
"""


def critique(llm, engine_result: Dict[str, Any], gender: str,
             fallback_confidence: float = 0.6) -> CriticOutput:
    payload = {
        "top_organs": [
            {
                "organ": o["organ"],
                "score": o["score"],
                "anchor_hits": o["anchor_hits"],
                "exact_sources": o["exact_sources"],
                "disease_sources": o["disease_sources"],
                "top_diseases": [d["disease"] for d in o["related_diseases"][:3]],
            }
            for o in engine_result.get("top_organs", [])
        ],
        "top_diseases": [
            {
                "disease": d["disease"],
                "score": d["score"],
                "matched_organs": d["matched_organs"],
                "source_priorities": d["source_priorities"],
                "confirmed": bool(d.get("eleventh_lord_confirmation")),
            }
            for d in engine_result.get("top_diseases", [])
        ],
        "anchor_count": len(engine_result.get("anchors", [])),
        "gender": gender,
    }

    try:
        resp = llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(content=json.dumps(payload, default=str)),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        start, end = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[start:end + 1]) if start >= 0 else {}
        return CriticOutput(**parsed)
    except Exception as e:
        # Fail-safe: neutral critique so pipeline continues. The caller passes its own
        # confidence threshold, so a failed critic never triggers a re-score.
        return CriticOutput(
            issues=[f"critic_failed: {e}"],
            confidence=fallback_confidence,
            notes="Critic unavailable; using the confidence threshold so no re-score is triggered.",
        )