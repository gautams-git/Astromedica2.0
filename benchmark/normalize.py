"""Common ranking schema extracted from engine outputs. The engines themselves are not changed."""

from __future__ import annotations

from typing import Any


def unwrap_agentic(result: dict | None) -> tuple[dict, dict | None]:
    if not isinstance(result, dict):
        return {}, None
    if "engine_result" not in result:
        return result, None
    agent = {key: value for key, value in result.items() if key != "engine_result"}
    engine = result.get("engine_result") or {}
    if not isinstance(engine, dict):
        engine = {"value": engine}
    return engine, agent


def _label(item: dict) -> str:
    return str(item.get("label") or item.get("organ") or item.get("disease") or item.get("system") or "")


def _score(item: dict) -> float | None:
    value = item.get("score")
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _public_trail(trail) -> list[dict[str, Any]]:
    """Keep the lines that already add up to a group score. Drop anything that is not JSON."""
    if not isinstance(trail, list):
        return []
    rows = []
    for step in trail:
        if not isinstance(step, dict):
            continue
        rows.append({
            "priority": step.get("priority") or "",
            "source_label": step.get("source_label") or "",
            "planet": step.get("planet") or "",
            "layer": step.get("layer") or "",
            "evidence_term": step.get("evidence_term") or "",
            "anchor_priority": step.get("anchor_priority") or "",
            "anchor_label": step.get("anchor_label") or "",
            "anchor_term": step.get("anchor_term") or "",
            "match_type": step.get("match_type") or "",
            "weight": step.get("weight"),
        })
    return rows


def rankings_from_engine(engine: dict | None) -> list[dict[str, Any]]:
    if not isinstance(engine, dict):
        return []
    rows = (
        engine.get("all_groups")
        or engine.get("all_organs")
        or engine.get("all_diseases")
        or engine.get("top4")
        or engine.get("top_organs")
        or engine.get("top_diseases")
        or engine.get("top3")
        or []
    )
    if not isinstance(rows, list):
        return []
    output = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        organs = item.get("matched_organs") or item.get("organs") or []
        diseases = item.get("matched_diseases") or item.get("diseases") or []
        if item.get("organ") and not organs:
            organs = [item["organ"]]
        if item.get("disease") and not diseases:
            diseases = [item["disease"]]
        output.append({
            "label": _label(item),
            "score": _score(item),
            "organs": list(organs) if isinstance(organs, list) else [],
            "diseases": list(diseases) if isinstance(diseases, list) else [],
            "trail": _public_trail(item.get("trail")),
        })
    return output


def fallback_note(engine: dict | None) -> str:
    if not isinstance(engine, dict):
        return ""
    return str(engine.get("fallback_note") or "")


def confidence_from_agent(agent: dict | None) -> float | None:
    if not isinstance(agent, dict):
        return None
    value = agent.get("confidence")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
