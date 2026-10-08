"""
Combined Algo top organs, reshaped into the hitlist cluster format that
chat_module / symptom_mapper already understand ({"top4": [cluster, ...]},
each cluster with label/score/hitlists/convergence/trail). Feeding the chat
this instead of processed_logic.json's hitlist v1 makes it ask about the
Combined Algo's top 3 organs without changing its question/scoring logic.

Uses the v1 Combined Algo (algorithm_comparison.py) — the one the /results
page shows — so the chat, RAG and the RL report all match that page.
"""

from __future__ import annotations

import json

CHAT_PLAN_SOURCE = "combined_algo"
TOP_N = 3


def compute_combined_algo_summary(folder: str) -> dict:
    # Recomputed from processed_logic.json rather than read from the saved
    # algorithm_comparison.json, which carries a chart-specific rank swap the
    # /results page doesn't apply (patient 009: saved Anus-first, page Heart-first).
    from algorithm_comparison import build_combined_algo_summary

    with open(f"{folder}/processed_logic.json", encoding="utf-8") as f:
        processed = json.load(f)
    return build_combined_algo_summary(processed)


def combined_algo_as_hitlist(summary: dict, top_n: int = TOP_N) -> dict:
    clusters = []
    for entry in (summary.get("top_organs") or [])[:top_n]:
        organ = entry.get("organ", "")
        # "Rashi" / "Nakshatra" in the source string is how symptom_mapper
        # tells organs from diseases when it builds question branches.
        trail = [{"term": organ, "source": "Combined Rashi organ"}]
        trail += [
            {"term": d.get("disease", ""), "source": "Combined Nakshatra disease"}
            for d in (entry.get("top_diseases") or [])
            if d.get("disease")
        ]
        clusters.append({
            "label": organ,
            "score": entry.get("consensus_score", 0),
            "hitlists": entry.get("supporting_algorithms", []),
            "convergence": entry.get("algorithm_support_count", 0),
            "trail": trail,
        })
    return {"top4": clusters}


def load_combined_algo_hitlist(folder: str) -> dict:
    return combined_algo_as_hitlist(compute_combined_algo_summary(folder))
