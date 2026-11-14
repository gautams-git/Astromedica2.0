"""Bounded RAG reads. Never contributes scores. Never writes."""
import asyncio
from typing import Any, Dict, List

from agentic_chat import config as cfg


async def fetch(term: str, context: Dict[str, Any],
                retriever) -> Dict[str, Any]:
    """Return {'hits': [...]} or {} on failure/timeout/off."""
    if cfg.RAG_MODE == "off":
        return {}
    try:
        hits: List[Dict[str, Any]] = await asyncio.wait_for(
            retriever.search(query=term, k=3, filters=context),
            timeout=cfg.RAG_TIMEOUT,
        )
        if not hits:
            return {}
        return {"hits": hits}
    except Exception as exc:
        return {"error": type(exc).__name__}


def examples_from_hits(hits: List[Dict[str, Any]], limit: int = 3) -> str:
    lines = []
    for i, h in enumerate(hits[:limit], 1):
        text = str(h.get("answer", h.get("text", ""))).strip()
        label = h.get("label", "")
        if text:
            lines.append(f"  {i}. [{label or 'example'}] {text[:160]}")
    return "\n".join(lines) if lines else "(no examples)"