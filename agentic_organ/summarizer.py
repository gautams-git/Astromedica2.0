"""Summarizer node for the organ engine."""
import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage


SUMMARIZER_SYSTEM = """You are a medical-astrology EXPLAINER for the
ORGAN ↔ DISEASE correlation engine.

Write a clear, cautious 3-5 sentence summary for a user.

HARD RULES:
- ONLY mention organs and diseases that appear in the provided data.
- Do NOT invent new terms.
- Do NOT contradict the ranking.
- Reference WHY the top organ ranked first (cite 2-3 anchor/source hits
  and one disease link).
- If the critic flagged low support or ambiguity, mention it honestly.
- End with an encouraging, non-alarmist note.
- NEVER give medical advice. NEVER prescribe.
"""


def summarize(llm, engine_result: Dict[str, Any], critique: Dict[str, Any],
              gender: str) -> str:
    payload = {
        "top_organs": [
            {
                "organ": o["organ"],
                "score": o["score"],
                "anchor_hits": o["anchor_hits"],
                "exact_sources": o["exact_sources"],
                "related_diseases": [
                    {"disease": d["disease"], "score": d["score"]}
                    for d in o["related_diseases"][:3]
                ],
                "trail": o.get("trail", [])[:5],
            }
            for o in engine_result.get("top_organs", [])
        ],
        "top_diseases": engine_result.get("top_diseases", [])[:5],
        "critic_notes": critique.get("notes", ""),
        "critic_issues": critique.get("issues", []),
        "gender": gender,
    }
    try:
        resp = llm.invoke([
            SystemMessage(content=SUMMARIZER_SYSTEM),
            HumanMessage(content=json.dumps(payload, default=str)),
        ])
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return f"(Summary unavailable: {e}. See organ/disease trail data.)"