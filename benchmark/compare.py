"""Cross-track differences. A deviation is either explained or marked unexplained."""

from __future__ import annotations

from typing import Any


def _labels(rankings: list[dict]) -> list[str]:
    return [item.get("label") or "" for item in rankings or []]


def _scores(rankings: list[dict]) -> dict[str, float | None]:
    return {item.get("label") or "": item.get("score") for item in rankings or []}


def compare_tracks(left: dict | None, right: dict | None, left_name: str, right_name: str) -> dict[str, Any]:
    left = left or {}
    right = right or {}
    if left.get("error") and not left.get("rankings"):
        return {
            "pair": f"{left_name}_vs_{right_name}",
            "status": "not_compared",
            "explanation": left.get("error") or f"{left_name} did not produce a ranking.",
        }
    if right.get("error") and not right.get("rankings"):
        return {
            "pair": f"{left_name}_vs_{right_name}",
            "status": "not_compared",
            "explanation": right.get("error") or f"{right_name} did not produce a ranking.",
        }

    left_labels = _labels(left.get("rankings"))
    right_labels = _labels(right.get("rankings"))
    left_scores = _scores(left.get("rankings"))
    right_scores = _scores(right.get("rankings"))
    added = [label for label in right_labels if label not in left_scores]
    removed = [label for label in left_labels if label not in right_scores]
    score_deltas = []
    for label in left_labels:
        if label in right_scores and left_scores[label] is not None and right_scores[label] is not None:
            delta = round(float(right_scores[label]) - float(left_scores[label]), 4)
            if abs(delta) > 0.02:
                score_deltas.append({"label": label, "delta": delta, "from": left_scores[label], "to": right_scores[label]})

    ranking_changed = left_labels != right_labels
    fallback_changed = (left.get("fallback_note") or "") != (right.get("fallback_note") or "")
    confidence_changed = left.get("confidence") != right.get("confidence")
    changed = bool(ranking_changed or score_deltas or added or removed or fallback_changed)

    explanation = "No material score, ranking, or fallback difference."
    classification = "equivalent"
    if changed:
        audit = right.get("weight_audit") or {}
        weight_changes = (audit.get("validation") or {}).get("changed") or audit.get("ablation") or []
        if weight_changes:
            explanation = "Scores or rankings moved after a validated weight change. See ablation for the key responsible."
            classification = "explained_weight_change"
        elif (left.get("agent") or {}).get("profile_used") != (right.get("agent") or {}).get("profile_used"):
            explanation = "The agent selected a different profile from the original engine."
            classification = "explained_profile_change"
        elif right.get("agent") and not left.get("agent"):
            explanation = "Agentic orchestration returned a different ranking while the recorded weights stayed on the engine default. The cause is not a weight edit."
            classification = "unexplained" if score_deltas or ranking_changed else "equivalent"
            if classification == "unexplained":
                explanation = "Track difference remains after comparing default weights. Classified as unexplained rather than attributed to a weight."
        else:
            explanation = "A difference is present and is not traced to a weight or profile change."
            classification = "unexplained"

    return {
        "pair": f"{left_name}_vs_{right_name}",
        "status": classification,
        "explanation": explanation,
        "ranking_changed": ranking_changed,
        "left_order": left_labels[:8],
        "right_order": right_labels[:8],
        "added": added[:8],
        "removed": removed[:8],
        "score_deltas": score_deltas[:8],
        "fallback_changed": fallback_changed,
        "confidence_changed": confidence_changed,
    }
