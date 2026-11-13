from __future__ import annotations

from typing import Any


def _normalize(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _extract_hitlist_diseases(group: dict) -> list[str]:
    diseases: list[str] = []
    seen: set[str] = set()
    for row in group.get("trail", []) or []:
        source = str(row.get("source", ""))
        term = str(row.get("term", "")).strip()
        if not term:
            continue
        if "HL2" not in source:
            continue
        key = _normalize(term)
        if key in seen:
            continue
        seen.add(key)
        diseases.append(term)
    return diseases


def _extract_hitlist_organs(group: dict) -> list[str]:
    organs: list[str] = []
    seen: set[str] = set()
    for row in group.get("trail", []) or []:
        source = str(row.get("source", ""))
        term = str(row.get("term", "")).strip()
        if not term:
            continue
        if "HL3" not in source:
            continue
        key = _normalize(term)
        if key in seen:
            continue
        seen.add(key)
        organs.append(term)
    return organs


def build_common_findings(hitlist: dict | None, rashi_correlation: dict | None) -> dict:
    hitlist = hitlist or {}
    rashi_correlation = rashi_correlation or {}

    hit_groups = hitlist.get("top4") or []
    rashi_groups = rashi_correlation.get("top3") or []

    rashi_by_label = {
        _normalize(group.get("label", "")): group
        for group in rashi_groups
        if group.get("label")
    }

    cards = []
    for hit_group in hit_groups:
        label = str(hit_group.get("label", "")).strip()
        if not label:
            continue

        rashi_group = rashi_by_label.get(_normalize(label))
        if not rashi_group:
            continue

        hit_diseases = _extract_hitlist_diseases(hit_group)
        rashi_diseases = list(rashi_group.get("matched_diseases", []) or [])
        rashi_organs = list(rashi_group.get("matched_organs", []) or [])
        hit_organs = _extract_hitlist_organs(hit_group)

        hit_disease_map = {_normalize(item): item for item in hit_diseases if _normalize(item)}
        rashi_disease_map = {_normalize(item): item for item in rashi_diseases if _normalize(item)}
        common_disease_keys = [key for key in hit_disease_map if key in rashi_disease_map]

        hit_organ_map = {_normalize(item): item for item in hit_organs if _normalize(item)}
        rashi_organ_map = {_normalize(item): item for item in rashi_organs if _normalize(item)}
        common_organ_keys = [key for key in hit_organ_map if key in rashi_organ_map]

        cards.append({
            "label": label,
            "hitlist_score": round(float(hit_group.get("score", 0.0) or 0.0), 1),
            "rashi_score": round(float(rashi_group.get("score", 0.0) or 0.0), 1),
            "shared_match_count": len(common_disease_keys),
            "support_summary": "Hit List + Rashi Correlation",
            "common_diseases": [hit_disease_map[key] for key in common_disease_keys],
            "common_organs": [hit_organ_map[key] for key in common_organ_keys],
            "hitlist_diseases": hit_diseases,
            "rashi_diseases": rashi_diseases,
            "matched_anchors": list(rashi_group.get("matched_anchors", []) or []),
            "supporting_priorities": list(rashi_group.get("supporting_priorities", []) or []),
            "has_exact_common_disease": bool(common_disease_keys),
        })

    cards.sort(
        key=lambda item: (
            item["shared_match_count"],
            item["hitlist_score"] + item["rashi_score"],
        ),
        reverse=True,
    )

    return {
        "title": "Common Groups & Diseases",
        "subtitle": "Shared findings between Hit List and Rashi Correlation",
        "common_group_count": len(cards),
        "cards": cards,
    }
